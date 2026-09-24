"""Teacher-labeled daily decisions: soft route and risk probabilities from a frontier model.

Jev is described as training toward reference probabilities averaged over frontier models.
``corpus/teacher_daily_*.jsonl`` holds tasks written and labeled by one frontier model (the
coding assistant building this repo), so it is a single-teacher approximation of that
signal. Each row: ``t`` task, ``w`` active window, ``r`` route probabilities, ``k`` P(risky).
Tasks are split into train/dev by a hash of their text, and each training task is shown in
several presentations so the model learns the task, not one option layout.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .benchmarks import daily_state
from .daily_synthetic import DESCRIPTIONS, RISK_QUESTIONS, ROUTE_QUESTIONS, ROUTES
from .data import DecisionExample, Question, write_jsonl

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def load_teacher_rows(corpus: str | Path = CORPUS) -> list[dict]:
    rows = []
    for path in sorted(Path(corpus).glob("teacher_daily_*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{path.name}:{number}"
            unknown = set(row["r"]) - set(ROUTES)
            if unknown:
                raise ValueError(f"{where}: unknown routes {sorted(unknown)}")
            total = sum(row["r"].values())
            if abs(total - 1.0) > 0.02 or min(row["r"].values()) <= 0:
                raise ValueError(f"{where}: route probabilities must be positive and sum to 1")
            if not 0.0 <= row["k"] <= 1.0:
                raise ValueError(f"{where}: risk probability outside [0, 1]")
            row["r"] = {name: value / total for name, value in row["r"].items()}
            rows.append(row)
    return rows


def _is_dev(task: str) -> bool:
    return int(hashlib.sha256(task.casefold().encode()).hexdigest()[:8], 16) % 100 < 15


def _presentation(rng: random.Random, row: dict, split: str, index: int) -> DecisionExample:
    weighted = [name for name in ROUTES if name in row["r"]]
    others = [name for name in ROUTES if name not in row["r"]]
    shown = weighted + rng.sample(others, rng.randint(max(0, 3 - len(weighted)), len(others)))
    rng.shuffle(shown)
    described = rng.random() < 0.7
    options = [f"{n}: {rng.choice(DESCRIPTIONS[n])}" if described else n for n in shown]
    targets = [row["r"].get(name, 0.0) for name in shown]
    route = Question(
        rng.choice(ROUTE_QUESTIONS),
        options,
        max(range(len(shown)), key=targets.__getitem__),
        "choice",
        targets,
    )
    risk = Question(
        rng.choice(RISK_QUESTIONS),
        ["True", "False"],
        0 if row["k"] >= 0.5 else 1,
        "noul",
        [row["k"], 1.0 - row["k"]],
    )
    task = row["t"]
    if rng.random() < 0.5:
        task = task[0].upper() + task[1:]
    questions = [route, risk] if rng.random() < 0.5 else [risk, route]
    best = max(row["r"], key=row["r"].__getitem__)
    return DecisionExample(
        f"{split}-teacher-{index}",
        daily_state(task, row["w"]),
        questions,
        "teacher",
        best,
        split,
        metadata={"task": row["t"]},
    )


def generate_teacher_dataset(
    output_dir: str | Path, *, variants: int = 12, seed: int = 99, corpus: str | Path = CORPUS
) -> dict:
    """Dev keeps one canonical presentation per held-out task; train gets ``variants`` each."""
    rows = load_teacher_rows(corpus)
    rng = random.Random(seed)
    train: list[DecisionExample] = []
    dev: list[DecisionExample] = []
    for row in rows:
        if _is_dev(row["t"]):
            dev.append(_presentation(rng, row, "dev", len(dev)))
        else:
            train.extend(_presentation(rng, row, "train", len(train)) for _ in range(variants))
    rng.shuffle(train)
    root = Path(output_dir)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "dev.jsonl", dev)
    manifest = {
        "seed": seed,
        "variants": variants,
        "tasks": {"train": len(rows) - len(dev), "dev": len(dev)},
        "splits": {
            split: {
                "rows": len(examples),
                "sha256": hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest(),
            }
            for split, examples in (("train", train), ("dev", dev))
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
