"""Live UI Automation check; runs only in a signed-in Windows desktop session."""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "nt" or os.environ.get("JEVLET_LIVE_DESKTOP") != "1",
    reason="set JEVLET_LIVE_DESKTOP=1 in an interactive Windows session",
)


def test_cached_reader_sees_foreground_window_quickly() -> None:
    from jevlet.desktop.native import NativeDesktop

    desktop = NativeDesktop()
    desktop.observe()  # warm the COM client
    started = time.perf_counter()
    context = desktop.observe()
    elapsed = time.perf_counter() - started
    assert context.window is not None
    assert context.controls, "foreground window exposed no accessible controls"
    assert all(control.rect is not None for control in context.controls)
    assert elapsed < 0.5
