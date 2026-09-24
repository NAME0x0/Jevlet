"""Native Windows 11 window treatment and a global hotkey, via DWM and user32."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
VK_SPACE = 0x20
WM_HOTKEY = 0x0312


def style_window(handle: int) -> None:
    """Dark title metrics, rounded corners, and the acrylic backdrop (Windows 11 22H2+)."""
    dwm = ctypes.windll.dwmapi
    for attribute, value in (
        (20, 1),  # DWMWA_USE_IMMERSIVE_DARK_MODE
        (33, 2),  # DWMWA_WINDOW_CORNER_PREFERENCE = round
        (38, 3),  # DWMWA_SYSTEMBACKDROP_TYPE = transient (acrylic)
    ):
        data = ctypes.c_int(value)
        dwm.DwmSetWindowAttribute(wintypes.HWND(handle), attribute, ctypes.byref(data), 4)

    class Margins(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_int),
            ("right", ctypes.c_int),
            ("top", ctypes.c_int),
            ("bottom", ctypes.c_int),
        ]

    margins = Margins(-1, -1, -1, -1)
    dwm.DwmExtendFrameIntoClientArea(wintypes.HWND(handle), ctypes.byref(margins))


def register_hotkey(handle: int, identifier: int, candidates: list[tuple[int, int, str]]) -> str:
    """Register the first free combination; returns its label or '' if all are taken."""
    user32 = ctypes.windll.user32
    for modifiers, key, label in candidates:
        if user32.RegisterHotKey(wintypes.HWND(handle), identifier, modifiers | MOD_NOREPEAT, key):
            return label
    return ""


def unregister_hotkey(handle: int, identifier: int) -> None:
    ctypes.windll.user32.UnregisterHotKey(wintypes.HWND(handle), identifier)


def is_hotkey_message(message: int) -> tuple[bool, int]:
    msg = wintypes.MSG.from_address(int(message))
    return msg.message == WM_HOTKEY, int(msg.wParam)


def bring_to_front(handle: int) -> None:
    """Foreground a window from a hotkey press (Windows allows it for the hotkey owner)."""
    user32 = ctypes.windll.user32
    user32.ShowWindow(wintypes.HWND(handle), 5)
    user32.SetForegroundWindow(wintypes.HWND(handle))
