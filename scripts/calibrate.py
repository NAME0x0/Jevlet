"""Fit per-question-kind temperatures on deployment-like data and store them in a checkpoint.

    python -m scripts.calibrate results/runs/jevlet-p-daily-v3/best.pt data/daily/current.pt \
        --data data/daily/dev.jsonl --data data/teacher/dev.jsonl

Never calibrate on the benchmark you report: it would make its calibration numbers meaningless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jevlet.calibration import fit_temperatures_by_kind
from jevlet.data import JsonlDecisionDataset
from jevlet.export import export_checkpoint
from jevlet.families import build_collator
from jevlet.training import collect_predictions, load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    device = resolve_device("auto")
    model, payload = load_checkpoint(args.checkpoint, device)
    collator = build_collator(model, payload.get("training_config", {}).get("data"))
    examples = [example for path in args.data for example in JsonlDecisionDataset(path).examples]
    loader = DataLoader(examples, batch_size=16, collate_fn=collator)
    logits, records, _ = collect_predictions(model, loader, device)
    temperatures = fit_temperatures_by_kind(logits, records)
    export_checkpoint(args.checkpoint, args.output, fp16=args.fp16)
    slim = torch.load(args.output, map_location="cpu", weights_only=True)
    slim["temperatures"] = temperatures
    slim["calibration"] = {
        "questions": len(records),
        "sources": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.data
        },
    }
    torch.save(slim, args.output)
    print(json.dumps({"temperatures": temperatures, "questions": len(records)}, indent=2))


if __name__ == "__main__":
    main()
