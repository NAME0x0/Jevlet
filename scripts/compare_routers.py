"""Compare a trained Jevlet checkpoint with the zero-shot semantic router on one split.

Both systems score the identical questions and options. Temperatures are fitted on the
same evaluated rows for both, so calibrated numbers are optimistic but comparable.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from jevlet.calibration import fit_temperature
from jevlet.data import JsonlDecisionDataset
from jevlet.families import build_collator
from jevlet.metrics import compute_metrics, metrics_by_family
from jevlet.training import collect_predictions, load_checkpoint, resolve_device


def _summary(logits, records) -> dict:
    temperature = fit_temperature(logits, records)
    overall = compute_metrics(logits, records)
    calibrated = compute_metrics(logits, records, temperature)
    return {
        "overall": overall,
        "calibrated_ece": calibrated["ece"],
        "calibrated_nll": calibrated["nll"],
        "temperature": temperature,
        "by_family": {
            family: {key: values[key] for key in ("questions", "accuracy", "ece", "nll")}
            for family, values in metrics_by_family(logits, records).items()
        },
    }


def _semantic_predictions(examples):
    from jevlet.semantic import Candidate, SemanticRouter

    router = SemanticRouter()
    logits, records = [], []
    started = time.perf_counter()
    for example in examples:
        for index, question in enumerate(example.questions):
            candidates = tuple(Candidate(option, option) for option in question.options)
            if len({candidate.name for candidate in candidates}) != len(candidates):
                continue
            values = router._logits(
                example.state, question.text, candidates, alpha=0.0, prototypes=()
            )
            logits.append(values.float())
            records.append(
                {
                    "family": example.family,
                    "label": question.label,
                    "target_probs": question.target_probs,
                    "is_ood": example.is_ood,
                    "is_unknown": question.is_unknown,
                    "question_index": index,
                }
            )
    elapsed = time.perf_counter() - started
    return logits, records, elapsed * 1000 / max(len(records), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()
    examples = JsonlDecisionDataset(args.data).examples[: args.limit]
    device = resolve_device("auto")
    model, payload = load_checkpoint(args.checkpoint, device)
    collator = build_collator(model, payload.get("training_config", {}).get("data", {}))
    loader = DataLoader(examples, args.batch_size, collate_fn=collator, num_workers=0)
    jevlet_logits, jevlet_records, speed = collect_predictions(model, loader, device)
    semantic_logits, semantic_records, semantic_ms = _semantic_predictions(examples)
    report = {
        "data": args.data,
        "examples": len(examples),
        "jevlet": {
            **_summary(jevlet_logits, jevlet_records),
            "latency_ms_per_question": speed["latency_ms_per_question"],
            "device": str(device),
        },
        "semantic_zero_shot": {
            **_summary(semantic_logits, semantic_records),
            "latency_ms_per_question": semantic_ms,
            "device": "cpu",
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=float)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
