"""Learning by demonstration: the user describes a task, then shows it with one click.

``ClickRecorder`` installs a low-level mouse hook, reports the next left click outside
Jevlet's own windows, and resolves the UI Automation element under it. The window's controls
are snapshotted when teaching starts, because a click may change the UI before it can be read.
Each demonstration becomes a labeled grounding decision (and a "click" skill label).
Everything stays in a local SQLite file; nothing is uploaded.
"""

from __future__ import annotations

import ctypes
import json
import queue
import sqlite3
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import asdict, dataclass
from pathlib import Path

from jevlet.benchmarks import daily_state
from jevlet.data import DecisionExample, Question
from jevlet.grounding import GROUNDING_QUESTION, actionable, control_text, grounding_options

WH_MOUSE_LL, WM_LBUTTONDOWN, WM_QUIT = 14, 0x0201, 0x0012
LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)


class _MouseInfo(ctypes.Structure):
    _fields_ = [
        ("pt", wintypes.POINT),
        ("mouse_data", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra", ctypes.c_size_t),
    ]


@dataclass(frozen=True, slots=True)
class Demonstration:
    command: str
    window_title: str
    process: str
    controls: tuple[dict, ...]
    target: dict


def root_window_at(x: int, y: int) -> int:
    user32 = ctypes.windll.user32
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
    handle = user32.WindowFromPoint(wintypes.POINT(x, y))
    return int(user32.GetAncestor(handle, 2) or 0)  # GA_ROOT


class ClickRecorder:
    """Report the next left click (screen coordinates) outside the ignored windows."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._clicks: queue.Queue[tuple[int, int]] = queue.Queue()

    def start(self) -> None:
        if self._thread is not None:
            return
        ready = threading.Event()

        def run() -> None:
            user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
            user32.CallNextHookEx.argtypes = [
                wintypes.HHOOK,
                ctypes.c_int,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.CallNextHookEx.restype = LRESULT
            user32.SetWindowsHookExW.restype = wintypes.HHOOK
            user32.SetWindowsHookExW.argtypes = [
                ctypes.c_int,
                HOOKPROC,
                wintypes.HINSTANCE,
                wintypes.DWORD,
            ]
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE

            @HOOKPROC
            def callback(code, message, data):
                if code == 0 and message == WM_LBUTTONDOWN:
                    info = ctypes.cast(data, ctypes.POINTER(_MouseInfo)).contents
                    self._clicks.put((info.pt.x, info.pt.y))  # never block inside the hook
                return user32.CallNextHookEx(None, code, message, data)

            self._callback = callback  # keep the trampoline alive
            self._thread_id = kernel32.GetCurrentThreadId()
            hook = user32.SetWindowsHookExW(
                WH_MOUSE_LL, callback, kernel32.GetModuleHandleW(None), 0
            )
            ready.set()
            message = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                pass
            user32.UnhookWindowsHookEx(hook)

        self._thread = threading.Thread(target=run, name="jevlet-click-hook", daemon=True)
        self._thread.start()
        ready.wait(2)

    def stop(self) -> None:
        if self._thread is None:
            return
        ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        self._thread.join(timeout=2)
        self._thread = None

    def next_click(
        self,
        ignore: Callable[[int], bool],
        timeout: float,
        cancel: threading.Event | None = None,
    ) -> tuple[int, int, int] | None:
        """Wait for a click on a window ``ignore`` rejects; returns (x, y, window) or None."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not (cancel and cancel.is_set()):
            try:
                x, y = self._clicks.get(timeout=0.1)
            except queue.Empty:
                continue
            window = root_window_at(x, y)
            if window and not ignore(window):
                return x, y, window
        return None


def _control_dict(element) -> dict:
    return {
        "name": element.name,
        "automation_id": element.automation_id,
        "control_type": element.control_type,
        "rect": list(element.rect),
        "enabled": element.enabled,
    }


def watch_for_demonstration(
    command: str,
    ignore: Callable[[int], bool],
    snapshot_window: int | None,
    timeout: float = 30.0,
    cancel: threading.Event | None = None,
) -> Demonstration | None:
    """Snapshot the target window, wait for one click, and describe what was clicked."""
    from jevlet.assistant.environment import _process_name
    from jevlet.desktop.native import NativeDesktop
    from jevlet.desktop.uia import FastUIA

    reader = FastUIA()
    before = reader.read(snapshot_window) if snapshot_window else ()
    recorder = ClickRecorder()
    recorder.start()
    try:
        click = recorder.next_click(ignore, timeout, cancel)
    finally:
        recorder.stop()
    if click is None:
        return None
    x, y, window = click
    target = reader.element_at(x, y)
    controls = before if window == snapshot_window and before else reader.read(window)
    info = NativeDesktop()._window_info(window)
    return Demonstration(
        command,
        info.title if info else "",
        _process_name(window),
        tuple(_control_dict(control) for control in controls),
        _control_dict(target),
    )


class DemonstrationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS demonstrations (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    command TEXT NOT NULL,
                    window_title TEXT NOT NULL,
                    process TEXT NOT NULL,
                    controls_json TEXT NOT NULL,
                    target_json TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def record(self, demonstration: Demonstration) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO demonstrations(command, window_title, process, controls_json,
                   target_json) VALUES (?, ?, ?, ?, ?)""",
                (
                    demonstration.command,
                    demonstration.window_title,
                    demonstration.process,
                    json.dumps(list(demonstration.controls), ensure_ascii=False),
                    json.dumps(demonstration.target, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def all(self) -> list[tuple[int, Demonstration]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, command, window_title, process, controls_json, target_json "
                "FROM demonstrations ORDER BY id"
            ).fetchall()
        return [
            (
                row[0],
                Demonstration(
                    row[1], row[2], row[3], tuple(json.loads(row[4])), json.loads(row[5])
                ),
            )
            for row in rows
        ]


@dataclass(frozen=True, slots=True)
class _Control:
    name: str
    automation_id: str
    control_type: str
    rect: tuple[int, ...] = ()
    enabled: bool = True


def demonstration_example(identifier: int, demonstration: Demonstration) -> DecisionExample | None:
    """A grounding decision whose label is the control the user actually clicked."""
    controls = actionable(
        [_Control(**{**c, "rect": tuple(c["rect"])}) for c in demonstration.controls]
    )
    target = demonstration.target
    target_text = control_text(target["control_type"], target["name"], target["automation_id"])
    options = grounding_options(controls)
    if target_text not in options:
        if target["control_type"] not in {c.control_type for c in controls} and not target["name"]:
            return None  # clicked an unlabeled surface: nothing learnable
        options.insert(len(options) - 1, target_text)
    return DecisionExample(
        f"demo-{identifier}",
        daily_state(demonstration.command, demonstration.window_title),
        [Question(GROUNDING_QUESTION, options, options.index(target_text))],
        "demonstration",
        demonstration.process or "unknown",
        "demonstration",
        metadata={"demonstration": asdict(demonstration)["target"]},
    )
