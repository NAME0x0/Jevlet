"""Fast Windows UI Automation reads: one cached cross-process call per window.

pywinauto fetches each property of each element with a separate COM round-trip. A
``CacheRequest`` asks UIA to return every needed property for the whole subtree in a single
``FindAllBuildCache`` call, which is what makes an always-on observer affordable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True, slots=True)
class UIElement:
    name: str
    automation_id: str
    control_type: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom in screen pixels
    enabled: bool


@lru_cache(maxsize=1)
def _client() -> tuple[Any, Any, dict[int, str]]:
    if os.name != "nt":
        raise RuntimeError("UI Automation requires Windows")
    import comtypes.client

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
    return automation, uia_module, type_names


class FastUIA:
    """Read the control view of one window with a single cached UIA query."""

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

    def read(self, handle: int, max_elements: int = 400) -> tuple[UIElement, ...]:
        root = self.automation.ElementFromHandle(handle)
        found = root.FindAllBuildCache(
            self.module.TreeScope_Descendants, self.condition, self.request
        )
        elements: list[UIElement] = []
        for index in range(found.Length):
            element = found.GetElement(index)
            if element.CachedIsOffscreen:
                continue
            box = element.CachedBoundingRectangle
            elements.append(
                UIElement(
                    element.CachedName or "",
                    element.CachedAutomationId or "",
                    self.type_names.get(element.CachedControlType, str(element.CachedControlType)),
                    (box.left, box.top, box.right, box.bottom),
                    bool(element.CachedIsEnabled),
                )
            )
            if len(elements) >= max_elements:
                break
        return tuple(elements)
