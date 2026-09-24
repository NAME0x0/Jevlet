from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.runner import run_successive_halving


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded Jevlet successive-halving research")
    parser.add_argument("--config", default="configs/overnight.json")
    parser.add_argument("--output", default="results/runs/overnight")
    parser.add_argument("--hours", type=float)
    parser.add_argument(
        "--extend-hours",
        type=float,
        help="Resume an interrupted run with a new deadline this many hours from now",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.hours is not None:
        config["research"]["hours"] = args.hours
    if args.extend_hours is not None:
        config["research"]["extend_hours"] = args.extend_hours
    summary = run_successive_halving(config, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
