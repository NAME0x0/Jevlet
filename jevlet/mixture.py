"""Combine decision datasets into one hash-pinned train/dev/vault mixture.

Each source directory holds ``train.jsonl``, ``dev.jsonl`` and ``vault.jsonl`` (or
``vault/vault.jsonl``). Rows keep their family/domain so metrics stay per-source. Vault rows
are written to a separate file that training never reads.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .data import DecisionExample, JsonlDecisionDataset, write_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_path(root: Path, split: str) -> Path | None:
    candidates = [root / f"{split}.jsonl"]
    if split == "vault":
        candidates.append(root / "vault" / "vault.jsonl")
    return next((path for path in candidates if path.exists()), None)


def build_mixture(
    sources: dict[str, str | Path],
    output_dir: str | Path,
    *,
    limits: dict[str, dict[str, int]] | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    """Sample up to ``limits[source][split]`` rows per source and split, then shuffle.

    Sampling is seeded per source and split, so adding a source does not change which rows
    another source contributes.
    """
    if not sources:
        raise ValueError("a mixture needs at least one source")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[DecisionExample]] = {"train": [], "dev": [], "vault": []}
    provenance: dict[str, Any] = {}
    for name, directory in sorted(sources.items()):
        root = Path(directory)
        provenance[name] = {"path": str(root), "splits": {}}
        for split in combined:
            path = _split_path(root, split)
            if path is None:
                if split == "train":
                    raise FileNotFoundError(f"{name} has no train.jsonl under {root}")
                continue
            examples = JsonlDecisionDataset(path).examples
            limit = (limits or {}).get(name, {}).get(split)
            if limit is not None and limit < len(examples):
                rng = random.Random(f"{seed}:{name}:{split}")
                examples = rng.sample(examples, limit)
            combined[split].extend(examples)
            provenance[name]["splits"][split] = {
                "source_sha256": _sha256(path),
                "rows": len(examples),
            }
    manifest: dict[str, Any] = {"seed": seed, "sources": provenance, "splits": {}}
    for split, examples in combined.items():
        if not examples:
            continue
        random.Random(f"{seed}:shuffle:{split}").shuffle(examples)
        path = output / ("vault" if split == "vault" else "") / f"{split}.jsonl"
        write_jsonl(path, examples)
        manifest["splits"][split] = {
            "rows": len(examples),
            "questions": sum(len(example.questions) for example in examples),
            "families": dict(sorted(Counter(example.family for example in examples).items())),
            "sha256": _sha256(path),
            "path": str(path.relative_to(output)),
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
