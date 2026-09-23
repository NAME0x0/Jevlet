"""Programmatically verifiable decision benchmark generation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .data import DecisionExample, Question, write_jsonl

FAMILIES = ("semantic", "rules", "missing", "contradiction", "permutation", "calibration")

TRAIN_INTENTS = {
    "billing": ("charged twice", "invoice is wrong", "refund has not arrived"),
    "technical": ("application crashes", "cannot sign in", "screen stays blank"),
    "sales": ("need a quote", "want an enterprise plan", "request a product demo"),
}
OOD_INTENTS = {
    "security": ("suspicious login", "credential was exposed", "report a phishing message"),
    "compliance": ("data retention request", "audit evidence needed", "privacy deletion request"),
    "operations": ("warehouse is delayed", "shipment routing failed", "inventory mismatch"),
}


def _semantic(rng: random.Random, split: str, index: int) -> DecisionExample:
    use_ood = split != "train" and rng.random() < (0.35 if split == "dev" else 0.75)
    intents = OOD_INTENTS if use_ood else TRAIN_INTENTS
    label_name = rng.choice(list(intents))
    utterance = rng.choice(intents[label_name])
    options = list(intents)
    rng.shuffle(options)
    question = Question(
        "Which queue should handle this request?", options, options.index(label_name)
    )
    return DecisionExample(
        f"{split}-semantic-{index}",
        f"Incoming request: {utterance}.",
        [question],
        "semantic",
        "support-ood" if use_ood else "support",
        split,
        use_ood,
    )


def _rules(rng: random.Random, split: str, index: int) -> DecisionExample:
    alpha, beta = rng.randint(0, 12), rng.randint(0, 12)
    threshold, ceiling = rng.randint(2, 8), rng.randint(6, 11)
    approved = alpha > threshold and beta <= ceiling
    state = (
        f"alpha is {alpha}. beta is {beta}. Approval requires alpha greater than {threshold} "
        f"AND beta no greater than {ceiling}."
    )
    first = Question("Is the request approved?", ["True", "False"], 0 if approved else 1, "noul")
    second_answer = not approved or alpha % 2 == 0
    second = Question(
        "Is either approval false OR alpha even?",
        ["True", "False"],
        0 if second_answer else 1,
        "noul",
    )
    return DecisionExample(
        f"{split}-rules-{index}", state, [first, second], "rules", "logic", split
    )


def _missing(rng: random.Random, split: str, index: int) -> DecisionExample:
    known = rng.choice(("red", "blue", "green"))
    entity = rng.choice(("unit", "sample", "package"))
    question = Question(
        f"Is the {entity} certified?",
        ["True", "False", "Unknown"],
        2,
        "noul",
        is_unknown=True,
    )
    return DecisionExample(
        f"{split}-missing-{index}",
        f"The {entity} color is {known}. No certification record is present.",
        [question],
        "missing",
        "incomplete-evidence",
        split,
    )


def _contradiction(rng: random.Random, split: str, index: int) -> DecisionExample:
    subject = rng.choice(("sensor", "account", "shipment", "device"))
    contradictory = rng.random() < 0.6
    if contradictory:
        state = f"The {subject} is active. The {subject} is not active."
        label = 1
    else:
        state = f"The {subject} is active. The {subject} is monitored."
        label = 0
    question = Question("Classify the evidence.", ["Consistent", "Contradictory", "Unknown"], label)
    return DecisionExample(
        f"{split}-contradiction-{index}",
        state,
        [question],
        "contradiction",
        "evidence",
        split,
    )


def _permutation(rng: random.Random, split: str, index: int) -> DecisionExample:
    workers = ["local worker", "Codex", "retrieval", "human"]
    selected = rng.choice(workers)
    options = list(workers)
    rng.shuffle(options)
    question = Question("Which worker does policy select?", options, options.index(selected))
    return DecisionExample(
        f"{split}-permutation-{index}",
        f"The routing policy explicitly selects {selected} for this task.",
        [question],
        "permutation",
        "routing",
        split,
        metadata={"canonical_label": selected},
    )


def _calibration(rng: random.Random, split: str, index: int) -> DecisionExample:
    red = rng.randint(1, 9)
    blue = rng.randint(1, 9)
    total = red + blue
    probabilities = [red / total, blue / total]
    question = Question(
        "What color is the next uniformly sampled token most likely to be?",
        ["red", "blue"],
        0 if red >= blue else 1,
        "choice",
        probabilities,
    )
    risk = min(4, max(0, round(4 * red / total)))
    score_probs = [0.0] * 5
    score_probs[risk] = 1.0
    score = Question(
        "Choose the ordered red-risk score.",
        ["1", "2", "3", "4", "5"],
        risk,
        "score",
        score_probs,
    )
    return DecisionExample(
        f"{split}-calibration-{index}",
        f"An urn contains {red} red tokens and {blue} blue tokens.",
        [question, score],
        "calibration",
        "known-probability",
        split,
    )


GENERATORS = {
    "semantic": _semantic,
    "rules": _rules,
    "missing": _missing,
    "contradiction": _contradiction,
    "permutation": _permutation,
    "calibration": _calibration,
}


def generate_examples(count: int, split: str, seed: int) -> list[DecisionExample]:
    rng = random.Random(seed)
    examples = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        examples.append(GENERATORS[family](rng, split, index))
    rng.shuffle(examples)
    return examples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_dataset(output_dir: str | Path, count: int = 80_000, seed: int = 1337) -> dict:
    root = Path(output_dir)
    split_counts = {
        "train": int(count * 0.80),
        "dev": int(count * 0.15),
        "vault": count - int(count * 0.80) - int(count * 0.15),
    }
    paths = {
        "train": root / "train.jsonl",
        "dev": root / "dev.jsonl",
        "vault": root / "vault" / "vault.jsonl",
    }
    manifest: dict = {"seed": seed, "requested_examples": count, "splits": {}}
    for offset, (split, split_count) in enumerate(split_counts.items()):
        examples = generate_examples(split_count, split, seed + offset * 10_000)
        write_jsonl(paths[split], examples)
        family_counts = Counter(example.family for example in examples)
        manifest["splits"][split] = {
            "count": len(examples),
            "questions": sum(len(example.questions) for example in examples),
            "families": dict(sorted(family_counts.items())),
            "sha256": _sha256(paths[split]),
            "path": str(paths[split].relative_to(root)),
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
