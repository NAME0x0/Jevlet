"""Fast Windows UI Automation: cached reads, element-at-point, and pattern-based activation.

pywinauto fetches each property of each element with a separate COM round-trip. A
``CacheRequest`` asks UIA to return every needed property for the whole subtree in a single
``FindAllBuildCache`` call, which is what makes an always-on observer affordable. COM objects
are apartment-bound, so each thread gets its own automation client.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class UIElement:
    name: str
    automation_id: str
    control_type: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom in screen pixels
    enabled: bool


_local = threading.local()


def _client() -> tuple[Any, Any, dict[int, str]]:
    if os.name != "nt":
        raise RuntimeError("UI Automation requires Windows")
    cached = getattr(_local, "client", None)
    if cached is not None:
        return cached
    import comtypes
    import comtypes.client

    comtypes.CoInitialize()
    comtypes.client.GetModule("UIAutomationCore.dll")
    from comtypes.gen import UIAutomationClient as uia_module

    automation = comtypes.client.CreateObject(
        uia_module.CUIAutomation, interface=uia_module.IUIAutomation
    )
    type_names = {
        getattr(uia_module, name): name[len("UIA_") : -len("ControlTypeId")]
        for name in dir(uia_module)
        if name.startswith("UIA_") and name.endswith("ControlTypeId")
    }
    _local.client = (automation, uia_module, type_names)
    return _local.client


class FastUIA:
    """Read and act on UI Automation elements with cached, batched queries."""

    def __init__(self) -> None:
        self.automation, self.module, self.type_names = _client()
        module = self.module
        request = self.automation.CreateCacheRequest()
        for property_id in (
            module.UIA_NamePropertyId,
            module.UIA_AutomationIdPropertyId,
            module.UIA_ControlTypePropertyId,
            module.UIA_BoundingRectanglePropertyId,
            module.UIA_IsEnabledPropertyId,
            module.UIA_IsOffscreenPropertyId,
        ):
            request.AddProperty(property_id)
        self.request = request
        self.condition = self.automation.ControlViewCondition

    def _element(self, raw) -> UIElement:
        box = raw.CachedBoundingRectangle
        return UIElement(
            raw.CachedName or "",
            raw.CachedAutomationId or "",
            self.type_names.get(raw.CachedControlType, str(raw.CachedControlType)),
            (box.left, box.top, box.right, box.bottom),
            bool(raw.CachedIsEnabled),
        )

    def _raw_elements(self, handle: int):
        root = self.automation.ElementFromHandle(handle)
        found = root.FindAllBuildCache(
            self.module.TreeScope_Descendants, self.condition, self.request
        )
        for index in range(found.Length):
            raw = found.GetElement(index)
            if not raw.CachedIsOffscreen:
                yield raw

    def read(self, handle: int, max_elements: int = 400) -> tuple[UIElement, ...]:
        elements: list[UIElement] = []
        for raw in self._raw_elements(handle):
            elements.append(self._element(raw))
            if len(elements) >= max_elements:
                break
        return tuple(elements)

    def element_at(self, x: int, y: int) -> UIElement:
        """The element under a screen point, e.g. where the user just clicked."""
        point = wintypes.POINT(x, y)
        raw = self.automation.ElementFromPointBuildCache(point, self.request)
        return self._element(raw)

    def activate(self, handle: int, control) -> str:
        """Invoke/select/toggle/expand a matching control; click its center as a fallback."""
        target = None
        for raw in self._raw_elements(handle):
            element = self._element(raw)
            if (
                element.control_type == control.control_type
                and element.name == (control.name or "")
                and (not control.automation_id or element.automation_id == control.automation_id)
            ):
                target = raw
                break
        if target is None:
            raise RuntimeError("the control is no longer on screen")
        module = self.module
        patterns = (
            (module.UIA_InvokePatternId, module.IUIAutomationInvokePattern, "Invoke", "clicked"),
            (
                module.UIA_SelectionItemPatternId,
                module.IUIAutomationSelectionItemPattern,
                "Select",
                "selected",
            ),
            (module.UIA_TogglePatternId, module.IUIAutomationTogglePattern, "Toggle", "toggled"),
            (
                module.UIA_ExpandCollapsePatternId,
                module.IUIAutomationExpandCollapsePattern,
                "Expand",
                "opened",
            ),
        )
        for pattern_id, interface, method, verb in patterns:
            try:
                pattern = target.GetCurrentPattern(pattern_id)
            except Exception:  # noqa: BLE001 - unsupported pattern
                continue
            if pattern:
                getattr(pattern.QueryInterface(interface), method)()
                return verb
        if self._element(target).control_type in {"Edit", "ComboBox"}:
            target.SetFocus()
            return "focused"
        left, top, right, bottom = self._element(target).rect
        _click((left + right) // 2, (top + bottom) // 2)
        return "clicked"


def _click(x: int, y: int) -> None:
    user32 = ctypes.windll.user32
    user32.SetCursorPos(x, y)
    time.sleep(0.03)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up
