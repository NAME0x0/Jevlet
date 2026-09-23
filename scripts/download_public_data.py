"""Download pinned public dataset slices into Jevlet's decision JSONL format."""

from __future__ import annotations

import argparse
import json

from jevlet.public_data import SPECS, download_public_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/public")
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(SPECS), default=["banking77", "boolq", "mnli"]
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--dev-limit", type=int, default=200)
    parser.add_argument("--vault-limit", type=int, default=200)
    parser.add_argument("--allow-unknown-license", action="store_true")
    args = parser.parse_args()
    manifest = download_public_data(
        args.output,
        args.datasets,
        seed=args.seed,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        vault_limit=args.vault_limit,
        allow_unknown_license=args.allow_unknown_license,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
