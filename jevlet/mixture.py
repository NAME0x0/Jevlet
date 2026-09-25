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
from typing import Any, NamedTuple

from .data import DecisionExample


class _Row(NamedTuple):
    line: str
    family: str
    questions: int


def _read_rows(path: Path) -> list[_Row]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            example = DecisionExample.from_dict(json.loads(raw))
            line = json.dumps(example.to_dict(), sort_keys=True) + "\n"
            rows.append(_Row(line, example.family, len(example.questions)))
    return rows


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
    # Rows are held as normalized JSON lines, not objects: a million-row mixture of objects
    # would need tens of gigabytes. The written bytes match ``write_jsonl`` exactly.
    combined: dict[str, list[_Row]] = {"train": [], "dev": [], "vault": []}
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
            rows = _read_rows(path)
            limit = (limits or {}).get(name, {}).get(split)
            if limit is not None and limit < len(rows):
                rng = random.Random(f"{seed}:{name}:{split}")
                rows = rng.sample(rows, limit)
            combined[split].extend(rows)
            provenance[name]["splits"][split] = {
                "source_sha256": _sha256(path),
                "rows": len(rows),
            }
    manifest: dict[str, Any] = {"seed": seed, "sources": provenance, "splits": {}}
    for split, rows in combined.items():
        if not rows:
            continue
        random.Random(f"{seed}:shuffle:{split}").shuffle(rows)
        path = output / ("vault" if split == "vault" else "") / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(row.line for row in rows)
        manifest["splits"][split] = {
            "rows": len(rows),
            "questions": sum(row.questions for row in rows),
            "families": dict(sorted(Counter(row.family for row in rows).items())),
            "sha256": _sha256(path),
            "path": str(path.relative_to(output)),
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
