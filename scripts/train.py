from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlet.training import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Jevlet experiment")
    parser.add_argument("--config", default="configs/proxy.json")
    parser.add_argument("--run-dir", default="results/runs/baseline")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    metrics = train_experiment(config, args.run_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
