from __future__ import annotations

import math

from jevlet.feedback import FeedbackStore
from jevlet.semantic import Candidate, SemanticRouter


class FakeEncoder:
    """Deterministic two-dimensional embedding fixture; no download needed."""

    def encode(self, texts, **kwargs):
        result = []
        for text in texts:
            if text.startswith("Request:"):
                result.append([1.0, 1.0])
            elif "code" in text:
                result.append([1.0, 0.0])
            else:
                result.append([0.0, 1.0])
        return result


class CountingEncoder(FakeEncoder):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return super().encode(texts, **kwargs)


OPTIONS = (Candidate("Codex", "write code"), Candidate("Human", "ask a human"))


def test_dynamic_options_and_unvalidated_gate() -> None:
    router = SemanticRouter(encoder=FakeEncoder())
    first = router.decision("task", "Who should act?", OPTIONS)
    swapped = router.decision("task", "Who should act?", OPTIONS[::-1])
    assert first.choice.options == ("Codex", "Human")
    assert swapped.choice.options == ("Human", "Codex")
    assert first.choice.selected == "Codex"
    assert swapped.choice.selected == "Human"
    assert first.gate == "verify"
    assert not first.calibrated
    assert math.isclose(sum(first.choice.probabilities), 1.0)


def test_typed_noul_and_score() -> None:
    router = SemanticRouter(encoder=FakeEncoder())
    noul = router.noul("some context", "Can this be determined?")
    score = router.score("some context", "Rate this", [1.0, 2.0, 3.0])
    assert noul.value in (True, False, None)
    assert math.isclose(
        noul.probability_true + noul.probability_false + noul.probability_unknown,
        1.0,
        abs_tol=1e-7,
    )
    assert 1.0 <= score.expectation <= 3.0


def test_high_cardinality_pruning_reuses_encoded_options() -> None:
    encoder = CountingEncoder()
    router = SemanticRouter(encoder=encoder)
    options = [Candidate(f"worker-{index}", f"task {index}") for index in range(20)]
    result = router.decision_two_stage("task", "Route this", options, top_k=4)
    assert len(result.choice.options) == 4
    assert result.gate == "verify"
    assert not result.calibrated
    assert encoder.calls == 1


def test_feedback_adapter_promotes_only_after_held_out_improvement(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    adapter_path = tmp_path / "adapter.json"
    router = SemanticRouter(encoder=FakeEncoder(), feedback_store=store)
    sparse = router.adapt(store, adapter_path)
    assert not sparse.promoted
    assert not adapter_path.exists()
    for index in range(25):
        identifier = store.log_decision(
            f"task {index}",
            "Who should act?",
            (("Codex", "write code"), ("Human", "ask a human")),
            "Codex",
        )
        store.submit_feedback(identifier, approved=False, corrected_option="Human")
    report = router.adapt(store, adapter_path)
    assert report.promoted
    assert report.candidate_nll < report.previous_nll
    restored = SemanticRouter(
        encoder=FakeEncoder(), feedback_store=store, adapter_path=adapter_path
    )
    result = restored.decision("new task", "Who should act?", OPTIONS)
    assert not result.calibrated  # Only five held-out labels: still require confirmation.
    assert result.gate == "verify"
    assert result.choice.selected == "Human"
