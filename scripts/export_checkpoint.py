"""Strip optimizer state (and optionally halve precision) for a deployable checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--fp16", action="store_true", help="Store weights in half precision")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = payload["model_state"]
    if args.fp16:
        state = {
            key: value.half() if value.is_floating_point() else value
            for key, value in state.items()
        }
    training = payload.get("training_config", {})
    slim = {
        "model_config": payload["model_config"],
        "model_state": state,
        "temperature": payload.get("temperature", 1.0),
        "metrics": payload.get("metrics", {}),
        "training_config": {"data": training.get("data", {}), "seed": training.get("seed")},
        "exported_from": str(Path(args.checkpoint).resolve()),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, destination)
    print(
        json.dumps(
            {"output": str(destination), "megabytes": destination.stat().st_size / 2**20},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
