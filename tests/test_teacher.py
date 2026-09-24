from __future__ import annotations

import json
import re

import pytest

from jevlet.benchmarks import CASES
from jevlet.teacher import generate_teacher_dataset, load_teacher_rows


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.casefold()))


def test_corpus_is_valid_and_covers_every_route() -> None:
    rows = load_teacher_rows()
    assert len(rows) >= 200
    best = {max(row["r"], key=row["r"].__getitem__) for row in rows}
    assert best == {"Codex", "Claude", "Gemini", "Retrieval", "Local", "Human"}
    assert len({row["t"].casefold() for row in rows}) == len(rows), "duplicate tasks"


def test_corpus_does_not_paraphrase_the_benchmark() -> None:
    for row in load_teacher_rows():
        for case in CASES:
            left, right = _tokens(row["t"]), _tokens(case.task)
            overlap = len(left & right) / len(left | right)
            assert overlap < 0.6, (row["t"], case.task)


def test_generated_targets_are_soft_and_normalized(tmp_path) -> None:
    manifest = generate_teacher_dataset(tmp_path, variants=3)
    assert manifest["tasks"]["dev"] > 0
    train_tasks, dev_tasks = set(), set()
    for split, bucket in (("train", train_tasks), ("dev", dev_tasks)):
        for line in (tmp_path / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            bucket.add(row["metadata"]["task"])
            for question in row["questions"]:
                assert sum(question["target_probs"]) == pytest.approx(1.0)
                assert question["target_probs"][question["label"]] == max(question["target_probs"])
    assert not train_tasks & dev_tasks
