"""Build a hash-pinned training mixture from several decision dataset directories."""

from __future__ import annotations

import argparse
import json

from jevlet.mixture import build_mixture


def _source(value: str) -> tuple[str, str, dict[str, int]]:
    """Parse ``name=path[:train,dev,vault]``; omitted caps mean all rows."""
    name, _, rest = value.partition("=")
    path, _, caps = rest.partition(":")
    if not name or not path:
        raise argparse.ArgumentTypeError("use name=path or name=path:train,dev,vault")
    limits: dict[str, int] = {}
    if caps:
        for split, cap in zip(("train", "dev", "vault"), caps.split(","), strict=False):
            if cap:
                limits[split] = int(cap)
    return name, path, limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--output", default="data/mixture")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    sources = {name: path for name, path, _ in args.source}
    limits = {name: caps for name, _, caps in args.source if caps}
    manifest = build_mixture(sources, args.output, limits=limits, seed=args.seed)
    print(json.dumps(manifest["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
