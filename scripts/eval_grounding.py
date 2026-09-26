"""Grounding accuracy split the ways an aggregate hides: control question vs risk question, and
target on screen vs foreign (the task belongs to another app; the answer is "None of these").

    python -m scripts.eval_grounding data/models/v6/jevlet-v6.pt data/grounding_v6/dev.jsonl
    python -m scripts.eval_grounding data/models/v6/jevlet-v6.pt data/grounding_v6/dev.jsonl \
        --only-ids-in data/mixture_v6/dev.jsonl    # the 3,000 rows the mixture dev set samples

Accuracy is argmax against the gold option. ECE is reported raw and with the checkpoint's
per-kind temperature. Prints one JSON object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jevlet.data import JsonlDecisionDataset
from jevlet.families import build_collator
from jevlet.metrics import compute_metrics
from jevlet.training import collect_predictions, load_checkpoint, resolve_device


def _block(logits: list[torch.Tensor], records: list[dict], temperature: float) -> dict:
    raw = compute_metrics(logits, records)
    calibrated = compute_metrics(logits, records, temperature)
    return {
        "questions": raw["questions"],
        "accuracy": raw["accuracy"],
        "ece": raw["ece"],
        "calibrated_ece": calibrated["ece"],
        "mean_confidence": calibrated["mean_confidence"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("--only-ids-in", help="keep only example ids that also occur in this jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = resolve_device("auto")
    model, payload = load_checkpoint(args.checkpoint, device)
    temperatures = payload.get("temperatures") or {}
    examples = JsonlDecisionDataset(args.data).examples
    if args.only_ids_in:
        keep = {
            json.loads(line)["example_id"]
            for line in Path(args.only_ids_in).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        examples = [example for example in examples if example.example_id in keep]
    foreign = {e.example_id: bool((e.metadata or {}).get("foreign")) for e in examples}
    collator = build_collator(model, payload.get("training_config", {}).get("data"))
    loader = DataLoader(examples, batch_size=args.batch_size, collate_fn=collator)
    logits, records, _ = collect_predictions(model, loader, device)

    slices: dict[str, tuple[list, list]] = {}
    for logit, record in zip(logits, records, strict=True):
        if record["question_index"] == 0:
            where = (
                "foreign (answer: none)" if foreign[record["example_id"]] else "target on screen"
            )
            names = ("control: all", f"control: {where}")
        else:
            names = ("risk question",)
        for name in names:
            bucket = slices.setdefault(name, ([], []))
            bucket[0].append(logit.float().cpu())
            bucket[1].append(record)

    on_screen = slices.get("control: target on screen", ([], []))
    abstained = sum(
        int(logit.argmax()) == len(record["options"]) - 1
        for logit, record in zip(*on_screen, strict=True)
    )
    report = {
        "checkpoint": Path(args.checkpoint).name,
        "data": Path(args.data).name,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "examples": len(examples),
        "slices": {
            name: _block(
                slice_logits,
                slice_records,
                float(temperatures.get(slice_records[0]["kind"], payload.get("temperature", 1.0))),
            )
            for name, (slice_logits, slice_records) in sorted(slices.items())
        },
        # On-screen errors that answered "None of these controls" (the last option).
        "on_screen_answered_none": abstained / max(len(on_screen[1]), 1),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
