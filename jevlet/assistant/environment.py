"""What the assistant can act on, enumerated from this machine at runtime.

Nothing here is a fixed class list the model memorizes: installed apps, open windows, and
visible controls are read live and offered to the model as runtime options, which is the
same contract Jev's criteria follow.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from .skills import CATALOGUES

CACHE = Path(__file__).resolve().parents[2] / "data" / "apps_cache.json"
NOISE = ("uninstall", "help", "readme", "documentation", "release notes", "license", "website")


@dataclass(frozen=True, slots=True)
class App:
    name: str
    app_id: str


@dataclass(frozen=True, slots=True)
class Window:
    handle: int
    title: str
    process: str


# Catalogues live in shared/skills.json so the C# app and the trainer use the same lists.
SETTINGS_PAGES: dict[str, str] = dict(CATALOGUES["setting"])
KNOWN_FOLDERS: dict[str, str] = dict(CATALOGUES["folder"])
WEBSITES: dict[str, str] = dict(CATALOGUES["website"])
# name -> (modifiers, key); resolved to virtual-key codes by the executor.
SHORTCUTS: dict[str, tuple[tuple[str, ...], str]] = {
    name: (tuple(modifiers), key) for name, (modifiers, key) in CATALOGUES["shortcut"].items()
}


def _start_apps() -> list[App]:
    command = "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"
    output = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout
    rows = json.loads(output or "[]")
    apps: dict[str, App] = {}
    for row in rows if isinstance(rows, list) else [rows]:
        name, app_id = str(row.get("Name", "")).strip(), str(row.get("AppID", "")).strip()
        if name and app_id and not any(word in name.casefold() for word in NOISE):
            apps.setdefault(name.casefold(), App(name, app_id))
    return sorted(apps.values(), key=lambda app: app.name.casefold())


def installed_apps(max_age_hours: float = 24.0, refresh: bool = False) -> list[App]:
    """Start-menu apps (Win32 and Store), cached because enumeration takes ~1.5 s."""
    if (
        not refresh
        and CACHE.exists()
        and time.time() - CACHE.stat().st_mtime < max_age_hours * 3600
    ):
        return [App(**row) for row in json.loads(CACHE.read_text(encoding="utf-8"))]
    if os.name != "nt":
        return []
    apps = _start_apps()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps([{"name": a.name, "app_id": a.app_id} for a in apps]), encoding="utf-8"
    )
    return apps


def _process_name(handle: int) -> str:
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(wintypes.HWND(handle), ctypes.byref(pid))
    process = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
    if not process:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buffer))
        if kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
            return Path(buffer.value).stem
        return ""
    finally:
        kernel32.CloseHandle(process)


def open_windows(exclude_handles: tuple[int, ...] = ()) -> list[Window]:
    """Visible, titled, top-level windows in z-order (most recent first)."""
    if os.name != "nt":
        return []
    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    windows: list[Window] = []

    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    dwmapi = ctypes.windll.dwmapi

    def alt_tab_candidate(handle) -> bool:
        """The same filter the Alt-Tab switcher uses: skips widgets, borders, and bars."""
        if not user32.IsWindowVisible(handle) or user32.GetWindow(handle, 4):  # GW_OWNER
            return False
        ex_style = user32.GetWindowLongW(handle, -20)  # GWL_EXSTYLE
        if ex_style & 0x80 and not ex_style & 0x40000:  # TOOLWINDOW without APPWINDOW
            return False
        cloaked = ctypes.c_int(0)
        dwmapi.DwmGetWindowAttribute(handle, 14, ctypes.byref(cloaked), 4)  # DWMWA_CLOAKED
        if cloaked.value:
            return False
        rect = wintypes.RECT()
        user32.GetWindowRect(handle, ctypes.byref(rect))
        return rect.right - rect.left > 40 and rect.bottom - rect.top > 40

    @callback_type
    def collect(handle, _parameter):
        if not alt_tab_candidate(handle):
            return True
        length = user32.GetWindowTextLengthW(handle)
        if not length or handle in exclude_handles:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, title, length + 1)
        text = title.value.strip()
        if text and text not in {"Program Manager", "Windows Input Experience"}:
            windows.append(Window(int(handle), text, _process_name(int(handle))))
        return True

    user32.EnumWindows(collect, 0)
    return windows
