from __future__ import annotations

import argparse
import json

from jevlet.synthetic import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed synthetic Jevlet benchmark")
    parser.add_argument("--output", default="data")
    parser.add_argument("--count", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    manifest = generate_dataset(args.output, args.count, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
