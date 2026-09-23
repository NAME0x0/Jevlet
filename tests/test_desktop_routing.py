from __future__ import annotations

import pytest

from jevlet.api import Choice, Noul
from jevlet.benchmarks import ROUTE_QUESTION
from jevlet.desktop.native import DesktopContext, WindowInfo
from jevlet.desktop.routing import DesktopRoutingSession
from jevlet.feedback import FeedbackStore
from jevlet.semantic import RouteDecision


class FakeRouter:
    def decision(self, state, question, candidates):
        assert question == ROUTE_QUESTION
        assert "Fix the failing test" in state and "Editor" in state
        assert candidates[0].description != candidates[0].name
        choice = Choice(tuple(c.name for c in candidates), (0.7, 0.3), candidates[0].name, 0.7)
        return RouteDecision(choice, "verify", False)


class FakeSystemOne:
    calibrated = True

    def evaluate(self, state, questions):
        assert set(questions) == {"route", "risk"}
        names = tuple(questions["route"].criteria)
        return {
            "route": Choice(names, (0.99, 0.01), names[0], 0.99),
            "risk": Noul(0.8, 0.2, None, True, 0.8),
        }

    def gate(self, confidence):
        return "execute" if confidence >= 0.9 else "verify"


def test_desktop_suggestion_and_corrected_feedback_reach_training_store(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    session = DesktopRoutingSession(FakeRouter(), store)
    context = DesktopContext(WindowInfo(42, "Editor", 123), ())
    options = ("Codex", "Human")

    decision = session.suggest("Fix the failing test", context, options)
    assert decision.choice.selected == "Codex"
    session.rate(-1, "Human")
    rows = store.labeled_decisions()
    assert len(rows) == 1
    assert rows[0].label == "Human"

    with pytest.raises(ValueError, match="Request a suggestion"):
        session.rate(1, None)


def test_risky_task_never_auto_executes_even_at_high_route_confidence(tmp_path) -> None:
    session = DesktopRoutingSession(FakeSystemOne(), FeedbackStore(tmp_path / "f.sqlite3"))
    context = DesktopContext(WindowInfo(7, "Bank portal", 1), ())
    decision = session.suggest("Wire the deposit", context, ("Local", "Human"))
    assert decision.risk == pytest.approx(0.8)
    assert decision.gate == "verify"
