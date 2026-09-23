from __future__ import annotations

import argparse
import json

from torch.utils.data import DataLoader

from jevlet.data import JsonlDecisionDataset
from jevlet.families import build_collator
from jevlet.metrics import compute_metrics
from jevlet.training import collect_predictions, load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Jevlet checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=("dev", "vault"), default="dev")
    parser.add_argument("--unlock-vault", action="store_true")
    parser.add_argument("--data", help="Override the evaluation JSONL (vault still needs unlock)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.split == "vault" and not args.unlock_vault:
        raise SystemExit("vault evaluation requires the explicit --unlock-vault flag")
    device = resolve_device("auto")
    model, payload = load_checkpoint(args.checkpoint, device)
    training_config = payload.get("training_config", {})
    data_config = training_config.get("data", {})
    data_path = data_config.get("dev", "data/dev.jsonl")
    if args.split == "vault":
        data_path = data_config.get("vault", "data/vault/vault.jsonl")
    if args.data:
        data_path = args.data
    dataset = JsonlDecisionDataset(data_path)
    collator = build_collator(model, data_config)
    loader = DataLoader(dataset, args.batch_size, collate_fn=collator, num_workers=0)
    logits, records, speed = collect_predictions(model, loader, device)
    metrics = compute_metrics(logits, records, float(payload.get("temperature", 1.0)))
    print(json.dumps({**metrics, **speed, "split": args.split}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
