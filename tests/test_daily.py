from __future__ import annotations

import json
import random

import pytest

from jevlet.benchmarks import CASES, ROUTES, daily_driver_examples
from jevlet.daily_synthetic import _example, generate_daily_dataset


def test_generator_is_deterministic_and_well_formed(tmp_path) -> None:
    first = generate_daily_dataset(tmp_path / "a", counts=(300, 30, 30), seed=5)
    second = generate_daily_dataset(tmp_path / "b", counts=(300, 30, 30), seed=5)
    assert first["splits"]["train"]["sha256"] == second["splits"]["train"]["sha256"]
    rows = [json.loads(line) for line in (tmp_path / "a" / "train.jsonl").read_text().splitlines()]
    for row in rows:
        route = next(q for q in row["questions"] if q["kind"] == "choice")
        gold = route["options"][route["label"]].split(":")[0]
        assert gold == row["domain"]
        assert len(route["options"]) >= 3
        if route["target_probs"] is not None:
            assert sum(route["target_probs"]) == pytest.approx(1.0)
            assert route["target_probs"][route["label"]] == max(route["target_probs"])
        if row["metadata"]["vague"]:
            assert row["domain"] == "Human" and route["is_unknown"]


def test_benchmark_sentences_never_appear_in_training_data() -> None:
    rng = random.Random(0)
    states = " ".join(_example(rng, "train", index).state.casefold() for index in range(4000))
    leaked = [case.task for case in CASES if case.task.casefold().rstrip(".?!") in states]
    assert not leaked


def test_benchmark_examples_cover_all_routes_and_put_task_in_state() -> None:
    examples = daily_driver_examples()
    assert {example.domain for example in examples} == set(ROUTES)
    for example, case in zip(examples, CASES, strict=True):
        assert case.task in example.state
        assert case.task not in example.questions[0].text
