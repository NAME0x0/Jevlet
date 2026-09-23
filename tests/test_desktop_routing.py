from __future__ import annotations

import pytest

from jevlet.api import Choice
from jevlet.desktop.native import DesktopContext, WindowInfo
from jevlet.desktop.routing import DesktopRoutingSession
from jevlet.feedback import FeedbackStore
from jevlet.semantic import RouteDecision


class FakeRouter:
    def decision(self, state, question, candidates):
        assert question == "Fix the failing test"
        assert "Editor" in state
        assert candidates[0].description != candidates[0].name
        choice = Choice(tuple(c.name for c in candidates), (0.7, 0.3), candidates[0].name, 0.7)
        return RouteDecision(choice, "verify", False)


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
