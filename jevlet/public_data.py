"""Versioned public text datasets converted to Jevlet decisions.

This module deliberately has no import-time Hugging Face dependency. Network access
occurs only when ``download_public_data`` is called.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from .data import DecisionExample, Question, write_jsonl


@dataclass(frozen=True)
class PublicDatasetSpec:
    name: str
    repo: str
    train_split: str
    vault_split: str
    license_note: str
    labels: tuple[str, ...] = ()
    max_options: int = 5
    # ``intent`` specs sample distractors from observed text labels (MASSIVE, CLINC150).
    task: str = "fixed"
    config: str | None = None
    label_column: str = "label"
    unknown_label: str | None = None
    question: str = ""


OUT_OF_SCOPE_OPTION = "none of these (out of scope)"


SPECS: dict[str, PublicDatasetSpec] = {
    "banking77": PublicDatasetSpec(
        "banking77",
        "mteb/banking77",
        "train",
        "test",
        "MTEB mirror says MIT; original PolyAI data says CC-BY-4.0.",
        max_options=4,
    ),
    "ag_news": PublicDatasetSpec(
        "ag_news",
        "fancyzhx/ag_news",
        "train",
        "test",
        "Dataset card lists license as unknown.",
        ("World", "Sports", "Business", "Science and technology"),
    ),
    "boolq": PublicDatasetSpec(
        "boolq",
        "google/boolq",
        "train",
        "validation",
        "CC-BY-SA-3.0.",
        ("No", "Yes"),
    ),
    "sst2": PublicDatasetSpec(
        "sst2",
        "stanfordnlp/sst2",
        "train",
        "validation",
        "Dataset card lists license as unknown.",
        ("Negative", "Positive"),
    ),
    "mnli": PublicDatasetSpec(
        "mnli",
        "nyu-mll/multi_nli",
        "train",
        "validation_mismatched",
        "Mixed CC-BY-3.0, CC-BY-SA-3.0, MIT, and public-domain terms.",
        ("Entailment", "Neutral", "Contradiction"),
    ),
    "yelp": PublicDatasetSpec(
        "yelp",
        "Yelp/yelp_review_full",
        "train",
        "test",
        "Dataset card lists license as other; review terms before redistribution.",
        ("1 star", "2 stars", "3 stars", "4 stars", "5 stars"),
    ),
    "massive": PublicDatasetSpec(
        "massive",
        "mteb/amazon_massive_intent",
        "train",
        "test",
        "MTEB mirror says Apache-2.0; original Amazon MASSIVE data says CC-BY-4.0.",
        max_options=6,
        task="intent",
        config="en",
        question="Which assistant action does the user want?",
    ),
    "clinc": PublicDatasetSpec(
        "clinc",
        "clinc/clinc_oos",
        "train",
        "test",
        "CC-BY-3.0.",
        max_options=6,
        task="intent",
        config="plus",
        label_column="intent",
        unknown_label="oos",
        question="Which supported intent matches the request, if any?",
    ),
}

UNKNOWN_LICENSE = frozenset({"ag_news", "sst2", "yelp"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(name: str, row: Mapping[str, Any]) -> str:
    if name == "boolq":
        content = (row["passage"], row["question"])
    elif name == "mnli":
        content = (row["premise"], row["hypothesis"])
    elif name == "sst2":
        content = (row["sentence"],)
    else:
        content = (row["text"],)
    normalized = "\n".join(" ".join(str(value).casefold().split()) for value in content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _banking_labels(rows: Iterable[Mapping[str, Any]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for row in rows:
        if "label_text" not in row:
            raise ValueError("mteb/banking77 row missing label_text")
        label = int(row["label"])
        text = str(row["label_text"]).replace("_", " ").strip()
        if label in mapping and mapping[label] != text:
            raise ValueError(f"inconsistent BANKING77 label {label}")
        mapping[label] = text
    if len(mapping) < 4:
        raise ValueError("BANKING77 sample needs at least four observed classes")
    return mapping


def _intent_name(row: Mapping[str, Any], spec: PublicDatasetSpec) -> str:
    raw = row.get("label_text", row.get(spec.label_column))
    return " ".join(str(raw).replace("_", " ").split()).casefold()


def _intent_vocabulary(rows: Iterable[Mapping[str, Any]], spec: PublicDatasetSpec) -> list[str]:
    unknown = spec.unknown_label.replace("_", " ") if spec.unknown_label else None
    names = sorted({_intent_name(row, spec) for row in rows} - {unknown})
    if len(names) < spec.max_options:
        raise ValueError(f"{spec.name} sample has too few intents for {spec.max_options} options")
    return names


def _convert_intent(
    spec: PublicDatasetSpec, row: Mapping[str, Any], split: str, seed: int, vocabulary: list[str]
) -> DecisionExample | None:
    name = _intent_name(row, spec)
    unknown = spec.unknown_label.replace("_", " ") if spec.unknown_label else None
    is_unknown = name == unknown
    if not is_unknown and name not in vocabulary:
        return None
    fingerprint = _identity(spec.name, row)
    rng = random.Random(seed ^ int(fingerprint[:16], 16))
    distractor_count = spec.max_options - (0 if is_unknown else 1)
    pool = [intent for intent in vocabulary if intent != name]
    options = rng.sample(pool, distractor_count) + ([] if is_unknown else [name])
    rng.shuffle(options)
    if unknown is not None:
        options.append(OUT_OF_SCOPE_OPTION)
    answer = OUT_OF_SCOPE_OPTION if is_unknown else name
    return DecisionExample(
        example_id=f"{spec.name}-{fingerprint[:20]}",
        state=str(row["text"]),
        questions=[Question(spec.question, options, options.index(answer), is_unknown=is_unknown)],
        family=f"public_{spec.name}",
        domain=spec.name,
        split=split,
        is_ood=False,
        metadata={"source_repo": spec.repo, "source_fingerprint": fingerprint},
    )


def convert_record(
    spec: PublicDatasetSpec,
    row: Mapping[str, Any],
    split: str,
    seed: int,
    banking_labels: Mapping[int, str] | None = None,
    intent_vocabulary: list[str] | None = None,
) -> DecisionExample | None:
    """Convert a labeled public row; return None for hidden/unlabeled test rows."""
    if spec.task == "intent":
        if intent_vocabulary is None:
            raise ValueError(f"{spec.name} conversion needs the intent vocabulary")
        return _convert_intent(spec, row, split, seed, intent_vocabulary)
    label = int(row["answer"] if spec.name == "boolq" else row["label"])
    if label < 0:
        return None
    if spec.name == "banking77":
        labels = banking_labels or {}
        if label not in labels:
            return None
        distractors = [key for key in sorted(labels) if key != label]
        rng = random.Random(seed ^ int(_identity(spec.name, row)[:16], 16))
        selected = [label, *rng.sample(distractors, min(spec.max_options - 1, len(distractors)))]
        rng.shuffle(selected)
        options = [labels[key] for key in selected]
        answer_index = selected.index(label)
    else:
        if label >= len(spec.labels):
            return None
        options = list(spec.labels)
        answer_index = label
    if spec.name == "boolq":
        state = str(row["passage"])
        question = str(row["question"])
    elif spec.name == "mnli":
        state = str(row["premise"])
        question = f"Relationship to: {row['hypothesis']}"
    elif spec.name == "sst2":
        state = str(row["sentence"])
        question = "What is the sentiment?"
    elif spec.name == "yelp":
        state = str(row["text"])
        question = "What star rating best matches the review?"
    elif spec.name == "ag_news":
        state = str(row["text"])
        question = "Which topic best matches the news item?"
    else:
        state = str(row["text"])
        question = "Which banking intent best matches the request?"
    fingerprint = _identity(spec.name, row)
    return DecisionExample(
        example_id=f"{spec.name}-{fingerprint[:20]}",
        state=state,
        questions=[Question(question, options, answer_index)],
        family=f"public_{spec.name}",
        domain=spec.name,
        split=split,
        is_ood=False,
        metadata={"source_repo": spec.repo, "source_fingerprint": fingerprint},
    )


def _default_info(repo: str) -> Mapping[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo)
    card = info.card_data
    return {"sha": info.sha, "license": card.get("license") if card else None}


def _default_loader(
    repo: str,
    split: str,
    revision: str,
    seed: int,
    config: str | None = None,
    label_column: str | None = None,
) -> Iterable[Mapping[str, Any]]:
    from datasets import ClassLabel, load_dataset

    # A bounded streaming shuffle can only see early rows of class-sorted sources
    # (notably BANKING77 and AG News). Materialize Arrow on disk, then globally
    # shuffle its indices; source_rows below remains bounded in Python memory.
    dataset = load_dataset(repo, config, split=split, revision=revision)
    feature = dataset.features.get(label_column) if label_column else None
    if isinstance(feature, ClassLabel) and "label_text" not in dataset.column_names:
        names = feature.names
        dataset = dataset.map(lambda row: {"label_text": names[row[label_column]]})
    return dataset.shuffle(seed=seed)


def download_public_data(
    output_dir: str | Path,
    names: Iterable[str] = ("banking77", "boolq", "mnli"),
    *,
    seed: int = 1337,
    train_limit: int = 1200,
    dev_limit: int = 200,
    vault_limit: int = 200,
    allow_unknown_license: bool = False,
    revisions: Mapping[str, str] | None = None,
    info_fn: Callable[[str], Mapping[str, Any]] = _default_info,
    loader_fn: Callable[..., Iterable[Mapping[str, Any]]] = _default_loader,
) -> dict[str, Any]:
    """Build immutable-by-hash train/dev/vault files from pinned source revisions.

    Vault labels are written separately and are never loaded by the training script.
    For stricter blindness, move vault files to another machine/account after generation.
    """
    selected = tuple(names)
    unknown = [name for name in selected if name not in SPECS]
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    restricted = [name for name in selected if name in UNKNOWN_LICENSE]
    if restricted and not allow_unknown_license:
        raise ValueError(
            f"license review required for {restricted}; pass allow_unknown_license=True to opt in"
        )
    if min(train_limit, dev_limit, vault_limit) < 1:
        raise ValueError("all split limits must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[DecisionExample]] = {"train": [], "dev": [], "vault": []}
    sources: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for name in selected:
        spec = SPECS[name]
        info = dict(info_fn(spec.repo))
        revision = str((revisions or {}).get(name) or info.get("sha") or "")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
            raise ValueError(f"{spec.repo} did not resolve to a pinned 40-digit revision")
        # Keep the in-memory Python sample bounded after Arrow's global shuffle.
        scan_limit = max(5000, 12 * (train_limit + dev_limit))
        extra = (
            {"config": spec.config, "label_column": spec.label_column}
            if spec.task == "intent"
            else {}
        )
        source_rows = list(
            islice(loader_fn(spec.repo, spec.train_split, revision, seed, **extra), scan_limit)
        )
        labels = _banking_labels(source_rows) if name == "banking77" else None
        vocabulary = _intent_vocabulary(source_rows, spec) if spec.task == "intent" else None
        counts = {"train": 0, "dev": 0, "vault": 0, "duplicates": 0, "invalid": 0}
        # Fixed hash partition prevents sampling order from changing split membership.
        for row in source_rows:
            fingerprint = _identity(name, row)
            split = "dev" if int(fingerprint[:8], 16) % 6 == 0 else "train"
            limit = dev_limit if split == "dev" else train_limit
            if counts[split] >= limit:
                continue
            if fingerprint in seen:
                counts["duplicates"] += 1
                continue
            example = convert_record(spec, row, split, seed, labels, vocabulary)
            if example is None:
                counts["invalid"] += 1
                continue
            seen.add(fingerprint)
            splits[split].append(example)
            counts[split] += 1
            if counts["train"] >= train_limit and counts["dev"] >= dev_limit:
                break
        for row in loader_fn(spec.repo, spec.vault_split, revision, seed + 1, **extra):
            if counts["vault"] >= vault_limit:
                break
            fingerprint = _identity(name, row)
            if fingerprint in seen:
                counts["duplicates"] += 1
                continue
            example = convert_record(spec, row, "vault", seed, labels, vocabulary)
            if example is None:
                counts["invalid"] += 1
                continue
            seen.add(fingerprint)
            splits["vault"].append(example)
            counts["vault"] += 1
        if min(counts["train"], counts["dev"], counts["vault"]) == 0:
            raise RuntimeError(f"{name} produced an empty split: {counts}")
        sources[name] = {
            "repo": spec.repo,
            "revision": revision,
            "source_splits": {"train_dev": spec.train_split, "vault": spec.vault_split},
            "reported_license": info.get("license"),
            "license_note": spec.license_note,
            "counts": counts,
        }
    hashes = {}
    for split, examples in splits.items():
        path = output / f"{split}.jsonl"
        write_jsonl(path, examples)
        hashes[split] = {"rows": len(examples), "sha256": _sha256(path), "path": str(path)}
    manifest = {"seed": seed, "sources": sources, "splits": hashes}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
