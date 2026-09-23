from __future__ import annotations

import pytest

from jevlet.api import Choice, Noul, Score
from jevlet.synthetic import generate_dataset
from jevlet.system_one import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOne,
    option_text,
)
from jevlet.training import train_experiment


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> SystemOne:
    root = tmp_path_factory.mktemp("system_one")
    generate_dataset(root / "data", count=36, seed=4)
    config = {
        "seed": 4,
        "device": "cpu",
        "model": {
            "max_seq_len": 384,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "ffn_mult": 2,
            "dropout": 0.0,
            "gradient_checkpointing": False,
        },
        "data": {"train": str(root / "data/train.jsonl"), "dev": str(root / "data/dev.jsonl")},
        "training": {"batch_size": 2, "max_steps": 1, "fp16": False},
        "evaluation": {"batch_size": 4, "permutation_examples": 0, "benchmark_repeats": 1},
    }
    train_experiment(config, root / "run")
    return SystemOne(str(root / "run" / "best.pt"), device="cpu")


def test_one_packed_call_equals_separate_calls(engine) -> None:
    state = "Ticket: the invoice total is wrong and the customer is angry."
    questions = {
        "queue": ChoiceQuestion("Which queue?", {"billing": "Money issues", "tech": "Bugs"}),
        "urgent": NoulQuestion("Is this urgent?"),
        "severity": ScoreQuestion("How severe?", ["Minor", "Major", "Blocking"]),
    }

    def distribution(answer):
        if isinstance(answer, Noul):
            return (answer.probability_true, answer.probability_false)
        return answer.probabilities

    together = engine.evaluate(state, questions)
    for identifier, question in questions.items():
        alone = engine.evaluate(state, {identifier: question})[identifier]
        assert distribution(together[identifier]) == pytest.approx(distribution(alone), abs=1e-5)
    assert isinstance(together["queue"], Choice) and together["queue"].selected in {
        "billing",
        "tech",
    }
    assert isinstance(together["urgent"], Noul) and together["urgent"].probability_unknown is None
    assert isinstance(together["severity"], Score) and 1.0 <= together["severity"].expectation <= 3


def test_cardinality_and_level_limits(engine) -> None:
    with pytest.raises(ValueError, match="at most 255"):
        engine.choice("s", "pick", [f"option {index}" for index in range(256)])
    with pytest.raises(ValueError, match="2-10"):
        engine.evaluate("s", {"q": ScoreQuestion("rate", ["only"])})
    with pytest.raises(ValueError, match="uniquely"):
        engine.choice("s", "pick", ["same", "same"])


def test_ungated_until_user_validated(engine) -> None:
    decision = engine.decision("state", "route", ["Codex", "Human"])
    assert decision.gate == "verify" and not decision.calibrated


def test_slim_fp16_export_loads_and_matches(engine, tmp_path, monkeypatch) -> None:
    import sys

    from scripts import export_checkpoint

    output = tmp_path / "slim.pt"
    monkeypatch.setattr(sys, "argv", ["export", engine.checkpoint_path, str(output), "--fp16"])
    export_checkpoint.main()
    slim = SystemOne(str(output), device="cpu")
    question = {"q": ChoiceQuestion("Which queue?", ["billing", "tech", "sales"])}
    expected = engine.evaluate("A refund is missing.", question)["q"].probabilities
    actual = slim.evaluate("A refund is missing.", question)["q"].probabilities
    assert actual == pytest.approx(expected, abs=1e-2)


def test_option_text_uses_description_only_when_informative() -> None:
    assert option_text("Codex", "Codex") == "Codex"
    assert option_text("Codex", "Edit code") == "Codex: Edit code"
