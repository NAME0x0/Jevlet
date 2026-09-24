"""Strip optimizer state (and optionally halve precision) for a deployable checkpoint."""

from __future__ import annotations

import argparse
import json

from jevlet.export import export_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--fp16", action="store_true", help="Store weights in half precision")
    args = parser.parse_args()
    target = export_checkpoint(args.checkpoint, args.output, fp16=args.fp16)
    print(json.dumps({"output": str(target), "megabytes": target.stat().st_size / 2**20}))


if __name__ == "__main__":
    main()
