"""Fine-tune the daily-driver checkpoint on your rated decisions; promote only if it helps.

    python -m scripts.personalize --checkpoint data/daily/current.pt

Roll back one promotion by copying ``previous.pt`` over ``current.pt``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from jevlet.feedback import FeedbackStore
from jevlet.personalize import personalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/daily/current.pt")
    parser.add_argument("--db", default="data/feedback.sqlite3")
    parser.add_argument("--replay", default="data/mixture_v4/dev.jsonl")
    parser.add_argument("--demonstrations", default="data/demonstrations.sqlite3")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = personalize(
        args.checkpoint,
        FeedbackStore(args.db),
        args.replay,
        steps=args.steps,
        device=args.device,
        demonstrations=Path(args.demonstrations),
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
