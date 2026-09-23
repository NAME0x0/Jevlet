"""Small, local Windows desktop adapter. No browser driver or remote service."""

from __future__ import annotations

import ctypes
import os
import subprocess
from ctypes import wintypes
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowInfo:
    handle: int
    title: str
    process_id: int


@dataclass(frozen=True, slots=True)
class ControlInfo:
    name: str
    automation_id: str
    control_type: str
    rect: tuple[int, int, int, int] | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class DesktopContext:
    window: WindowInfo | None
    controls: tuple[ControlInfo, ...]


class NativeDesktop:
    """Operate only in the logged-in Windows session, at the caller's privilege level."""

    APPS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "explorer": "explorer.exe",
    }

    @staticmethod
    def _user32():
        if os.name != "nt":
            raise RuntimeError("Desktop control requires an interactive Windows session")
        return ctypes.windll.user32

    def _window_info(self, handle: int) -> WindowInfo | None:
        if not handle:
            return None
        user32 = self._user32()
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        title = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(handle) + 1)
        user32.GetWindowTextW(handle, title, len(title))
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(process_id))
        return WindowInfo(handle, title.value, process_id.value)

    def active_window(self) -> WindowInfo | None:
        user32 = self._user32()
        user32.GetForegroundWindow.restype = wintypes.HWND
        return self._window_info(user32.GetForegroundWindow())

    def list_windows(self) -> tuple[WindowInfo, ...]:
        user32 = self._user32()
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        windows: list[WindowInfo] = []

        @callback_type
        def collect(handle, _parameter):
            if user32.IsWindowVisible(handle):
                info = self._window_info(handle)
                if info and info.title.strip():
                    windows.append(info)
            return True

        user32.EnumWindows(collect, 0)
        return tuple(windows)

    _fast_reader = None

    def observe(self, max_controls: int = 80) -> DesktopContext:
        window = self.active_window()
        if window is None:
            return DesktopContext(None, ())
        try:
            # One cached UIA query (~80 ms) instead of per-property calls (~600 ms).
            from .uia import FastUIA

            if NativeDesktop._fast_reader is None:
                NativeDesktop._fast_reader = FastUIA()
            elements = NativeDesktop._fast_reader.read(window.handle, max_controls)
            controls = tuple(
                ControlInfo(e.name, e.automation_id, e.control_type, e.rect, e.enabled)
                for e in elements
            )
            return DesktopContext(window, controls)
        except Exception:  # noqa: BLE001 - any COM failure falls back to pywinauto below
            pass
        try:
            from pywinauto import Desktop
        except ImportError:
            return DesktopContext(window, ())
        try:
            root = Desktop(backend="uia").window(handle=window.handle)
            controls = tuple(
                ControlInfo(
                    element.element_info.name or "",
                    element.element_info.automation_id or "",
                    element.element_info.control_type or "",
                )
                for element in root.descendants()[:max_controls]
            )
        except (OSError, RuntimeError, ValueError):
            controls = ()
        return DesktopContext(window, controls)

    @staticmethod
    def capture_screen():
        """Return a PIL image; nothing is saved or uploaded."""
        from PIL import ImageGrab

        try:
            return ImageGrab.grab(all_screens=True)
        except OSError as exc:
            raise RuntimeError(
                "Screen capture is unavailable here. "
                "Run the panel in your signed-in desktop session."
            ) from exc

    def launch_app(self, app: str) -> None:
        try:
            executable = self.APPS[app]
        except KeyError as exc:
            raise ValueError(f"App is not allowlisted: {app}") from exc
        subprocess.Popen([executable], shell=False)

    def switch_window(self, title: str) -> None:
        matches = [
            window for window in self.list_windows() if title.casefold() in window.title.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one matching window, found {len(matches)}")
        self.activate_window(matches[0].handle)

    def activate_window(self, handle: int) -> None:
        user32 = self._user32()
        user32.IsWindow.argtypes = [wintypes.HWND]
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.GetForegroundWindow.restype = wintypes.HWND
        if not user32.IsWindow(handle):
            raise RuntimeError("Target window closed after preview")
        user32.ShowWindow(handle, 5)
        if not user32.SetForegroundWindow(handle):
            raise RuntimeError("Windows declined to switch the foreground window")
        if user32.GetForegroundWindow() != handle:
            raise RuntimeError("Target window is not in the foreground")

    def focus_control(self, automation_id: str) -> None:
        if not automation_id:
            raise ValueError("An automation ID is required")
        try:
            from pywinauto import Desktop
        except ImportError as exc:
            raise RuntimeError("Install pywinauto for accessible-control focus") from exc
        window = self.active_window()
        if window is None:
            raise RuntimeError("No active window")
        matches = (
            Desktop(backend="uia").window(handle=window.handle).descendants(auto_id=automation_id)
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one matching control, found {len(matches)}")
        matches[0].set_focus()

    def type_text(self, value: str) -> None:
        """Send UTF-16 key events to the current foreground control."""
        if not value or len(value) > 4000:
            raise ValueError("Text must contain 1–4000 characters")
        user32 = self._user32()
        pointer_size = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else wintypes.DWORD

        class KeyInput(ctypes.Structure):
            _fields_ = [
                ("virtual_key", wintypes.WORD),
                ("scan_code", wintypes.WORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("extra_info", pointer_size),
            ]

        class InputValue(ctypes.Union):
            _fields_ = [("keyboard", KeyInput), ("padding", ctypes.c_byte * 32)]

        class Input(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("value", InputValue)]

        user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        user32.SendInput.restype = wintypes.UINT
        encoded = value.encode("utf-16-le")
        for offset in range(0, len(encoded), 128):
            units = [
                int.from_bytes(encoded[index : index + 2], "little")
                for index in range(offset, min(offset + 128, len(encoded)), 2)
            ]
            events = (Input * (len(units) * 2))()
            for index, unit in enumerate(units):
                events[2 * index].type = 1
                events[2 * index].value.keyboard = KeyInput(0, unit, 0x4, 0, 0)
                events[2 * index + 1].type = 1
                events[2 * index + 1].value.keyboard = KeyInput(0, unit, 0x6, 0, 0)
            count = user32.SendInput(len(events), events, ctypes.sizeof(Input))
            if count != len(events):
                raise RuntimeError(
                    "Windows blocked text input; elevated apps may require elevation"
                )
