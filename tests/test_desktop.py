from __future__ import annotations

import sqlite3

import pytest

from jevlet.desktop import ActionRequest, DesktopHarness, WindowInfo
from jevlet.desktop.feedback import DesktopFeedbackStore
from jevlet.desktop.native import DesktopContext


class FakeDesktop:
    APPS = {"notepad": "notepad.exe"}

    def __init__(self) -> None:
        self.window = WindowInfo(41, "Example", 2)
        self.calls: list[tuple[str, str | int]] = []
        self.ignore_activation = False

    def active_window(self) -> WindowInfo:
        return self.window

    def observe(self) -> DesktopContext:
        return DesktopContext(self.window, ())

    def list_windows(self) -> tuple[WindowInfo, ...]:
        return (WindowInfo(41, "Example", 2),)

    def activate_window(self, handle: int) -> None:
        self.calls.append(("activate", handle))
        if not self.ignore_activation:
            self.window = WindowInfo(handle, "Example", 2)

    def launch_app(self, app: str) -> None:
        self.calls.append(("launch", app))

    def switch_window(self, title: str) -> None:
        self.calls.append(("switch", title))

    def focus_control(self, automation_id: str) -> None:
        self.calls.append(("focus", automation_id))

    def type_text(self, value: str) -> None:
        self.calls.append(("type", value))


def test_action_requires_preview_and_explicit_confirmation() -> None:
    backend = FakeDesktop()
    harness = DesktopHarness(backend)
    proposal = harness.propose(ActionRequest("launch", "notepad"))
    with pytest.raises(PermissionError, match="confirmation"):
        harness.execute(proposal)
    assert backend.calls == []
    result = harness.execute(proposal, confirmed=True)
    assert result.success
    assert backend.calls == [("launch", "notepad")]
    with pytest.raises(PermissionError, match="stale"):
        harness.execute(proposal, confirmed=True)


def test_only_allowlisted_actions_and_apps() -> None:
    harness = DesktopHarness(FakeDesktop())
    with pytest.raises(ValueError, match="allowlisted"):
        harness.propose(ActionRequest("launch", "powershell"))
    with pytest.raises(ValueError, match="Unsupported"):
        harness.propose(ActionRequest("shell", "anything"))


def test_type_targets_the_previewed_window() -> None:
    backend = FakeDesktop()
    harness = DesktopHarness(backend)
    proposal = harness.propose(ActionRequest("type", text="hello"), expected_window_handle=41)
    backend.window = WindowInfo(99, "Jevlet", 3)
    harness.execute(proposal, confirmed=True)
    assert backend.calls == [("activate", 41), ("type", "hello")]


def test_type_aborts_if_foreground_does_not_match() -> None:
    backend = FakeDesktop()
    harness = DesktopHarness(backend)
    proposal = harness.propose(ActionRequest("type", text="hello"), expected_window_handle=41)
    backend.window = WindowInfo(99, "Different app", 3)
    backend.ignore_activation = True
    with pytest.raises(RuntimeError, match="Foreground window changed"):
        harness.execute(proposal, confirmed=True)
    assert backend.calls == [("activate", 41)]


def test_switch_uses_window_identified_at_preview() -> None:
    backend = FakeDesktop()
    harness = DesktopHarness(backend)
    proposal = harness.propose(ActionRequest("switch", "Exam"))
    assert proposal.expected_window_handle == 41
    harness.execute(proposal, confirmed=True)
    assert backend.calls == [("activate", 41)]


def test_switch_rejects_changed_target_after_preview() -> None:
    backend = FakeDesktop()
    harness = DesktopHarness(backend)
    proposal = harness.propose(ActionRequest("switch", "Exam"))
    backend.list_windows = lambda: (WindowInfo(41, "Different title", 2),)
    with pytest.raises(RuntimeError, match="changed after preview"):
        harness.execute(proposal, confirmed=True)
    assert backend.calls == []


def test_feedback_is_explicit_and_stores_no_typed_text(tmp_path) -> None:
    store = DesktopFeedbackStore(tmp_path / "desktop.sqlite3")
    store.rate("open notes", "Handle locally", "launch", 1)
    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT task, suggestion, action_kind, rating FROM desktop_feedback"
        ).fetchone()
    assert row == ("open notes", "Handle locally", "launch", 1)
    with pytest.raises(ValueError, match="thumbs"):
        store.rate("task", "route", "type", 0)
