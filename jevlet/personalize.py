"""Learn from the user's explicit feedback without letting one bad batch ship.

Labeled feedback (thumbs-up = the suggestion was right; thumbs-down with a correction = the
correction is right; a bare thumbs-down is never a label) is split by stable decision id:
60% fine-tuning, 20% temperature fitting, 20% promotion gate. The gate never sees rows the
temperature was fitted on. A candidate replaces the current model only if gate accuracy does
not fall, gate NLL improves, and accuracy on replayed general data drops by at most
``max_replay_drop``. The replaced model is kept as ``previous.pt`` for one-step rollback.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import DecisionExample, JsonlDecisionDataset, Question, write_jsonl
from .export import export_checkpoint
from .families import build_collator
from .feedback import FeedbackStore, LabeledDecision
from .metrics import compute_metrics
from .system_one import option_text
from .training import collect_predictions, load_checkpoint, resolve_device, train_experiment

CALIBRATION_MINIMUM = 20


@dataclass(frozen=True, slots=True)
class PersonalizationReport:
    promoted: bool
    reason: str
    counts: dict[str, int]
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)


def _example(row: LabeledDecision) -> DecisionExample:
    names = [name for name, _ in row.options]
    options = [option_text(name, description) for name, description in row.options]
    return DecisionExample(
        f"feedback-{row.id}",
        row.state,
        [Question(row.question, options, names.index(row.label))],
        "user_feedback",
        "feedback",
        "feedback",
    )


def _bucket(identifier: int) -> str:
    return "train" if identifier % 5 in (0, 1, 2) else "tune" if identifier % 5 == 3 else "gate"


def feedback_partitions(
    store: FeedbackStore, demonstrations: Path | None = None
) -> dict[str, list[DecisionExample]]:
    """Ratings and demonstrations, split by stable id so a row never changes partition."""
    parts: dict[str, list[DecisionExample]] = {"train": [], "tune": [], "gate": []}
    for row in store.labeled_decisions():
        parts[_bucket(row.id)].append(_example(row))
    if demonstrations is not None and demonstrations.exists():
        from .desktop.demonstrations import DemonstrationStore, demonstration_example

        for identifier, demo in DemonstrationStore(demonstrations).all():
            example = demonstration_example(identifier, demo)
            if example is not None:
                parts[_bucket(identifier)].append(example)
    return parts


def _score(checkpoint: Path, examples: list[DecisionExample], device: torch.device) -> dict:
    model, payload = load_checkpoint(checkpoint, device)
    collator = build_collator(model, payload.get("training_config", {}).get("data"))
    loader = DataLoader(examples, batch_size=16, collate_fn=collator)
    logits, records, _ = collect_predictions(model, loader, device)
    metrics = compute_metrics(logits, records, float(payload.get("temperature", 1.0)))
    return {"accuracy": float(metrics["accuracy"]), "nll": float(metrics["nll"])}


def validated_examples(checkpoint: str | Path) -> int:
    """How many held-out feedback rows the current model was last promoted on."""
    sidecar = Path(checkpoint).with_suffix(".json")
    if not sidecar.exists():
        return 0
    return int(json.loads(sidecar.read_text(encoding="utf-8")).get("validated_examples", 0))


def personalize(
    checkpoint: str | Path,
    store: FeedbackStore,
    replay_path: str | Path,
    *,
    steps: int = 150,
    replay_ratio: int = 3,
    replay_eval_size: int = 400,
    max_replay_drop: float = 0.01,
    minimum: tuple[int, int, int] = (8, 3, 3),
    learning_rate: float = 1e-4,
    backbone_learning_rate: float = 1e-5,
    device: str = "auto",
    seed: int = 1337,
    demonstrations: Path | None = None,
) -> PersonalizationReport:
    current = Path(checkpoint)
    parts = feedback_partitions(store, demonstrations)
    counts = {name: len(rows) for name, rows in parts.items()}
    if any(counts[name] < need for name, need in zip(parts, minimum, strict=True)):
        return PersonalizationReport(
            False, f"need at least {minimum} train/tune/gate labels", counts
        )
    rng = random.Random(seed)
    replay = list(JsonlDecisionDataset(replay_path).examples)
    rng.shuffle(replay)
    replay_eval = replay[:replay_eval_size]
    replay_train = replay[replay_eval_size:][: max(64, replay_ratio * counts["train"])]
    resolved = resolve_device(device)
    payload = torch.load(current, map_location="cpu", weights_only=True)
    with tempfile.TemporaryDirectory(prefix="jevlet_personalize_") as work:
        root = Path(work)
        train_rows = parts["train"] + replay_train
        rng.shuffle(train_rows)
        write_jsonl(root / "train.jsonl", train_rows)
        write_jsonl(root / "tune.jsonl", parts["tune"])
        config: dict[str, Any] = {
            "seed": seed,
            "device": str(resolved),
            "model": payload["model_config"],
            "data": {
                **payload.get("training_config", {}).get("data", {}),
                "train": str(root / "train.jsonl"),
                "dev": str(root / "tune.jsonl"),
            },
            "training": {
                "init_from": str(current),
                "batch_size": 8,
                "gradient_accumulation": 1,
                "max_steps": steps,
                "learning_rate": learning_rate,
                "backbone_learning_rate": backbone_learning_rate,
                "warmup_steps": min(10, steps),
                "loss": "ce_brier",
                "amp_dtype": "fp16" if resolved.type == "cuda" else "none",
            },
            "evaluation": {"batch_size": 16, "permutation_examples": 0, "benchmark_repeats": 1},
        }
        train_experiment(config, root / "run")
        candidate = root / "run" / "best.pt"
        before = {
            f"gate_{key}": value for key, value in _score(current, parts["gate"], resolved).items()
        }
        before["replay_accuracy"] = _score(current, replay_eval, resolved)["accuracy"]
        after = {
            f"gate_{key}": value
            for key, value in _score(candidate, parts["gate"], resolved).items()
        }
        after["replay_accuracy"] = _score(candidate, replay_eval, resolved)["accuracy"]
        if after["gate_accuracy"] < before["gate_accuracy"]:
            reason = "held-out feedback accuracy fell"
        elif after["gate_nll"] > before["gate_nll"] - 0.005:
            reason = "held-out feedback NLL did not improve"
        elif after["replay_accuracy"] < before["replay_accuracy"] - max_replay_drop:
            reason = "general-data accuracy dropped (forgetting)"
        else:
            reason = ""
        if reason:
            return PersonalizationReport(False, reason, counts, before, after)
        shutil.copy2(current, current.with_name("previous.pt"))
        export_checkpoint(candidate, current, fp16=True)
    report = PersonalizationReport(True, "promoted", counts, before, after)
    current.with_suffix(".json").write_text(
        json.dumps(
            {
                "validated_examples": counts["gate"],
                "calibrated": counts["gate"] >= CALIBRATION_MINIMUM,
                "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "report": asdict(report),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
