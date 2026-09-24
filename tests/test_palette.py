"""Keyboard contract of the palette, run headless (no window is shown on screen)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from jevlet.app.palette import Palette  # noqa: E402
from jevlet.assistant.planner import Plan  # noqa: E402
from jevlet.assistant.skills import SKILL_BY_KEY  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _palette(app, plan: Plan) -> tuple[Palette, list]:
    palette = Palette()
    palette.set_ready("test")
    events: list = []
    palette.run_requested.connect(lambda *args: events.append(("run", args[1].skill.key)))
    palette.replan_requested.connect(lambda *args: events.append(("replan", args[1].key)))
    palette.input.setText(plan.command)
    palette.request = 7
    palette.show_plan(7, plan, 1)
    return palette, events


def _press(palette: Palette, key) -> None:
    palette._key(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


def _plan(skill: str, confidence: float, risk: float, **args) -> Plan:
    plan = Plan("cmd", SKILL_BY_KEY[skill], confidence, risk, args)
    plan.arg_confidence = {slot: 0.95 for slot in args}
    plan.alternatives = [(SKILL_BY_KEY["website"], 0.03), (SKILL_BY_KEY["clarify"], 0.01)]
    return plan


def test_safe_confident_plan_runs_on_one_enter(app) -> None:
    palette, events = _palette(app, _plan("open_app", 0.97, 0.02, app="Spotify"))
    _press(palette, Qt.Key.Key_Return)
    assert events == [("run", "open_app")]


def test_risky_plan_needs_a_second_enter(app) -> None:
    palette, events = _palette(app, _plan("close_window", 0.97, 0.4, window="x"))
    _press(palette, Qt.Key.Key_Return)
    assert events == [] and palette.armed
    _press(palette, Qt.Key.Key_Return)
    assert events == [("run", "close_window")]


def test_escape_disarms_before_closing(app) -> None:
    palette, events = _palette(app, _plan("close_window", 0.97, 0.4, window="x"))
    _press(palette, Qt.Key.Key_Return)
    _press(palette, Qt.Key.Key_Escape)
    assert not palette.armed and events == []


def test_clarify_never_runs(app) -> None:
    palette, events = _palette(app, _plan("clarify", 0.9, 0.1))
    _press(palette, Qt.Key.Key_Return)
    _press(palette, Qt.Key.Key_Return)
    assert events == []


def test_arrow_then_enter_replans_the_alternative(app) -> None:
    palette, events = _palette(app, _plan("open_app", 0.97, 0.02, app="Spotify"))
    _press(palette, Qt.Key.Key_Down)
    assert palette.alternatives.isVisibleTo(palette)
    _press(palette, Qt.Key.Key_Return)
    assert events == [("replan", "website")]


def test_stale_plans_are_ignored(app) -> None:
    palette, _ = _palette(app, _plan("open_app", 0.97, 0.02, app="Spotify"))
    newer = _plan("volume", 0.9, 0.01, volume="Mute or unmute")
    palette.show_plan(6, newer, 1)  # an older request id arriving late
    assert palette.plan.skill.key == "open_app"
