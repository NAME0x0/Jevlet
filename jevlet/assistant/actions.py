"""Windows executors. Each returns an ``Outcome`` with an undo callable when one exists.

Only native mechanisms are used: ``SendInput`` for keys, window messages for window state,
the shell for launching, the registry for theme, and UI Automation patterns for controls.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
import urllib.parse
import winreg
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

VK = {
    "ctrl": 0x11, "shift": 0x10, "alt": 0x12, "win": 0x5B, "tab": 0x09, "enter": 0x0D,
    "f5": 0x74, "plus": 0xBB, "minus": 0xBD, "period": 0xBE, "left": 0x25, "right": 0x27,
    "play_pause": 0xB3, "next_track": 0xB0, "previous_track": 0xB1,
    "volume_up": 0xAF, "volume_down": 0xAE, "volume_mute": 0xAD,
}  # fmt: skip
EXTENDED = {0x5B, 0x25, 0x27, 0xB3, 0xB0, 0xB1, 0xAF, 0xAE, 0xAD}


@dataclass(frozen=True, slots=True)
class Outcome:
    ok: bool
    message: str
    undo: Callable[[], None] | None = None


class _KeyInput(ctypes.Structure):
    _fields_ = [
        ("vk", wintypes.WORD),
        ("scan", wintypes.WORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra", ctypes.c_size_t),
    ]


class _Union(ctypes.Union):
    _fields_ = [("ki", _KeyInput), ("padding", ctypes.c_byte * 32)]


class _Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _Union)]


def _send(events: list[tuple[int, bool]]) -> None:
    """Send (virtual_key, key_up) events in one atomic SendInput batch."""
    array = (_Input * len(events))()
    for index, (key, up) in enumerate(events):
        flags = (0x2 if up else 0) | (0x1 if key in EXTENDED else 0)
        array[index].type = 1
        array[index].u.ki = _KeyInput(key, 0, flags, 0, 0)
    sent = ctypes.windll.user32.SendInput(len(events), array, ctypes.sizeof(_Input))
    if sent != len(events):
        raise RuntimeError("Windows blocked the key press (an elevated window may have focus)")


def _key_code(name: str) -> int:
    if name in VK:
        return VK[name]
    if len(name) == 1 and name.isalnum():
        return ord(name.upper())
    raise ValueError(f"unknown key {name!r}")


def press(modifiers: tuple[str, ...], key: str, times: int = 1) -> None:
    codes = [_key_code(modifier) for modifier in modifiers]
    target = _key_code(key)
    for _ in range(times):
        events = [(code, False) for code in codes] + [(target, False), (target, True)]
        events += [(code, True) for code in reversed(codes)]
        _send(events)


def launch_app(app_id: str, name: str) -> Outcome:
    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
    return Outcome(True, f"Opened {name}")


def open_target(target: str, label: str) -> Outcome:
    os.startfile(target)  # noqa: S606 - URIs and shell folders only, never user paths
    return Outcome(True, f"Opened {label}")


def web_search(query: str) -> Outcome:
    os.startfile("https://www.google.com/search?q=" + urllib.parse.quote_plus(query))
    return Outcome(True, f"Searched the web for “{query}”")


def ask_ai(service: str, prompt: str) -> Outcome:
    encoded = urllib.parse.quote(prompt)
    urls = {
        "Claude": f"https://claude.ai/new?q={encoded}",
        "ChatGPT": f"https://chatgpt.com/?q={encoded}",
        "Gemini": "https://gemini.google.com/app",
    }
    set_clipboard(prompt)
    os.startfile(urls[service])
    return Outcome(True, f"Sent to {service} (also copied to clipboard)")


def set_clipboard(text: str) -> None:
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    data = text.encode("utf-16-le") + b"\x00\x00"
    handle = kernel32.GlobalAlloc(0x0002, len(data))
    pointer = kernel32.GlobalLock(handle)
    ctypes.memmove(pointer, data, len(data))
    kernel32.GlobalUnlock(handle)
    if not user32.OpenClipboard(None):
        raise RuntimeError("clipboard is busy")
    try:
        user32.EmptyClipboard()
        user32.SetClipboardData(13, handle)  # CF_UNICODETEXT
    finally:
        user32.CloseClipboard()


def _user32():
    user32 = ctypes.windll.user32
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsZoomed.argtypes = [wintypes.HWND]
    return user32


def focus_window(handle: int) -> bool:
    user32 = _user32()
    if user32.IsIconic(handle):
        user32.ShowWindow(handle, 9)  # SW_RESTORE
    # Tapping Alt releases Windows' foreground lock for this process.
    _send([(VK["alt"], False), (VK["alt"], True)])
    return bool(user32.SetForegroundWindow(handle))


def switch_to(handle: int, title: str, previous: int | None) -> Outcome:
    if not focus_window(handle):
        return Outcome(False, f"Windows refused to switch to {title}")
    undo = (lambda: focus_window(previous)) if previous else None
    return Outcome(True, f"Switched to {title}", undo)


def window_state(handle: int, title: str, action: str) -> Outcome:
    user32 = _user32()
    if action == "close":
        user32.PostMessageW(handle, 0x0010, 0, 0)  # WM_CLOSE: apps still ask to save work
        return Outcome(True, f"Asked {title} to close")
    was_zoomed = bool(user32.IsZoomed(handle))
    command = {"minimize": 6, "maximize": 3, "restore": 9}[action]
    user32.ShowWindow(handle, command)

    def undo() -> None:
        user32.ShowWindow(handle, 3 if was_zoomed else 9)

    verb = {"minimize": "Minimized", "maximize": "Maximized", "restore": "Restored"}[action]
    return Outcome(True, f"{verb} {title}", undo)


def media(action: str) -> Outcome:
    key = {"Play or pause": "play_pause", "Next track": "next_track"}.get(action, "previous_track")
    press((), key)
    return Outcome(True, action)


def volume(action: str, steps: int = 5) -> Outcome:
    if action == "Mute or unmute":
        press((), "volume_mute")
        return Outcome(True, "Toggled mute", lambda: press((), "volume_mute"))
    key, back = (
        ("volume_up", "volume_down") if action == "Volume up" else ("volume_down", "volume_up")
    )
    press((), key, steps)
    return Outcome(True, action, lambda: press((), back, steps))


THEME_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


def _theme_values() -> tuple[int, int]:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, THEME_KEY) as key:
        apps = winreg.QueryValueEx(key, "AppsUseLightTheme")[0]
        system = winreg.QueryValueEx(key, "SystemUsesLightTheme")[0]
    return int(apps), int(system)


def _write_theme(apps: int, system: int) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, THEME_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, "AppsUseLightTheme", 0, winreg.REG_DWORD, apps)
        winreg.SetValueEx(key, "SystemUsesLightTheme", 0, winreg.REG_DWORD, system)
    result = wintypes.DWORD()
    ctypes.windll.user32.SendMessageTimeoutW(
        0xFFFF, 0x001A, 0, "ImmersiveColorSet", 0x0002, 200, ctypes.byref(result)
    )


def theme(mode: str) -> Outcome:
    previous = _theme_values()
    value = 0 if mode == "Dark mode" else 1
    _write_theme(value, value)
    return Outcome(True, f"Switched to {mode.lower()}", lambda: _write_theme(*previous))


def type_text(handle: int | None, text: str) -> Outcome:
    from jevlet.desktop.native import NativeDesktop

    if handle:
        focus_window(handle)
        time.sleep(0.12)
    NativeDesktop().type_text(text)
    return Outcome(True, f"Typed {len(text)} characters")


def shortcut(name: str, keys: tuple[tuple[str, ...], str], handle: int | None) -> Outcome:
    if handle and "win" not in keys[0]:
        focus_window(handle)
        time.sleep(0.12)
    press(keys[0], keys[1])
    return Outcome(True, name)


def screenshot() -> Outcome:
    from PIL import ImageGrab

    folder = Path(os.path.expanduser("~")) / "Pictures" / "Screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / time.strftime("Jevlet %Y-%m-%d %H%M%S.png")
    ImageGrab.grab(all_screens=True).save(path)
    return Outcome(True, f"Saved {path.name}")


def lock() -> Outcome:
    ctypes.windll.user32.LockWorkStation()
    return Outcome(True, "Locked")


def click_control(handle: int, control) -> Outcome:
    from jevlet.desktop.uia import FastUIA

    focus_window(handle)
    time.sleep(0.1)
    how = FastUIA().activate(handle, control)
    label = control.name or control.automation_id
    return Outcome(True, f"{how.capitalize()} {control.control_type.lower()} “{label}”")
