from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlet.training import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Jevlet experiment")
    parser.add_argument("--config", default="configs/proxy.json")
    parser.add_argument("--run-dir", default="results/runs/baseline")
    parser.add_argument("--resume-from", help="Resume model, optimizer, RNG, and sampler state")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--amp-dtype", choices=("none", "fp16", "bf16"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.resume_from:
        config["training"]["resume_from"] = args.resume_from
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.amp_dtype:
        config["training"]["amp_dtype"] = args.amp_dtype
    metrics = train_experiment(config, args.run_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
