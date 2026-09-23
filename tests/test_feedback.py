from __future__ import annotations

import pytest

from jevlet.feedback import FeedbackStore


def test_feedback_requires_an_explicit_label_for_rejected_decisions(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    options = (("Codex", "write code"), ("Human", "ask a human"))
    rejected = store.log_decision("task", "Route?", options, "Codex")
    store.submit_feedback(rejected, approved=False)
    assert store.labeled_decisions() == []
    corrected = store.log_decision("task", "Route?", options, "Codex")
    store.submit_feedback(corrected, approved=False, corrected_option="Human")
    approved = store.log_decision("task", "Route?", options, "Human")
    store.submit_feedback(approved, approved=True)
    assert [(row.id, row.label) for row in store.labeled_decisions()] == [
        (corrected, "Human"),
        (approved, "Human"),
    ]


def test_feedback_rejects_invalid_or_duplicate_corrections(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    options = (("Codex", "write code"), ("Human", "ask a human"))
    identifier = store.log_decision("task", "Route?", options, "Codex")
    with pytest.raises(ValueError, match="existing option"):
        store.submit_feedback(identifier, approved=False, corrected_option="Unknown")
    with pytest.raises(ValueError, match="differ"):
        store.submit_feedback(identifier, approved=False, corrected_option="Codex")
    store.submit_feedback(identifier, approved=False, corrected_option="Human")
    with pytest.raises(ValueError, match="already"):
        store.submit_feedback(identifier, approved=True)
