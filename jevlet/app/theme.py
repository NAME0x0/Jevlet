"""Design tokens. Every color, font, size, and duration used by the UI is named here."""

from __future__ import annotations

import ctypes
import winreg
from dataclasses import dataclass


def system_accent() -> str:
    """The user's Windows accent color, so the palette reads as part of the OS."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\DWM") as key:
            abgr = int(winreg.QueryValueEx(key, "AccentColor")[0]) & 0xFFFFFFFF
        red, green, blue = abgr & 0xFF, (abgr >> 8) & 0xFF, (abgr >> 16) & 0xFF
        # Lift very dark accents so they stay legible on the dark surface.
        lift = max(0, 110 - max(red, green, blue))
        return f"#{min(255, red + lift):02X}{min(255, green + lift):02X}{min(255, blue + lift):02X}"
    except OSError:
        return "#5B9BD5"


def animations_enabled() -> bool:
    """Honors Settings > Accessibility > Visual effects > Animation effects."""
    enabled = ctypes.c_bool(True)
    ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0)
    return bool(enabled.value)


@dataclass(frozen=True)
class Tokens:
    accent: str
    surface: str = "rgba(22, 22, 26, 214)"
    surface_raised: str = "rgba(255, 255, 255, 10)"
    surface_hover: str = "rgba(255, 255, 255, 16)"
    surface_selected: str = "rgba(255, 255, 255, 24)"
    edge: str = "rgba(255, 255, 255, 22)"
    edge_highlight: str = "rgba(255, 255, 255, 36)"
    divider: str = "rgba(255, 255, 255, 14)"
    text: str = "#EDEDEF"
    text_secondary: str = "#A1A1AA"
    text_tertiary: str = "#71717A"
    warning: str = "#E8A23A"
    danger: str = "#E5484D"
    success: str = "#46A758"
    font: str = "Segoe UI Variable Text"
    font_display: str = "Segoe UI Variable Display"
    font_mono: str = "Cascadia Mono"
    size_input: int = 20
    size_title: int = 15
    size_body: int = 13
    size_meta: int = 11
    radius: int = 12
    radius_small: int = 6
    space_1: int = 4
    space_2: int = 8
    space_3: int = 12
    space_4: int = 16
    space_5: int = 20
    width: int = 640
    enter_ms: int = 140
    exit_ms: int = 90
    success_hold_ms: int = 1300


TOKENS = Tokens(accent=system_accent())
