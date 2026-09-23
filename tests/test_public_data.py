from __future__ import annotations

import json

import pytest

from jevlet.public_data import SPECS, convert_record, download_public_data
from notebooks.colab_runtime import CheckpointSync, restore_checkpoint


def test_banking_candidates_include_label_and_are_reproducible() -> None:
    row = {"text": "When will my card arrive?", "label": 2, "label_text": "card_arrival"}
    labels = {0: "get a refund", 1: "transfer money", 2: "card arrival", 3: "freeze account"}
    first = convert_record(SPECS["banking77"], row, "train", 42, labels)
    second = convert_record(SPECS["banking77"], row, "train", 42, labels)
    assert first is not None and second is not None
    assert first.to_dict() == second.to_dict()
    question = first.questions[0]
    assert len(question.options) == 4
    assert question.options[question.label] == "card arrival"


def test_public_split_is_disjoint_and_pinned(tmp_path) -> None:
    train_rows = [
        {
            "question": f"Does item {index} work?",
            "passage": f"Item {index} works.",
            "answer": bool(index % 2),
        }
        for index in range(100)
    ]
    vault_rows = [
        {
            "question": f"Does held-out item {index} work?",
            "passage": f"Held-out {index}.",
            "answer": bool(index % 2),
        }
        for index in range(10)
    ]

    def loader(repo, split, revision, seed):
        assert repo == "google/boolq"
        assert revision == "a" * 40
        return train_rows if split == "train" else vault_rows

    manifest = download_public_data(
        tmp_path,
        ["boolq"],
        train_limit=8,
        dev_limit=3,
        vault_limit=4,
        info_fn=lambda repo: {"sha": "a" * 40, "license": "cc-by-sa-3.0"},
        loader_fn=loader,
    )
    assert manifest["sources"]["boolq"]["revision"] == "a" * 40
    fingerprints = []
    for split in ("train", "dev", "vault"):
        rows = [json.loads(line) for line in (tmp_path / f"{split}.jsonl").read_text().splitlines()]
        assert len(rows) == manifest["splits"][split]["rows"]
        assert all(row["split"] == split for row in rows)
        fingerprints.extend(row["metadata"]["source_fingerprint"] for row in rows)
    assert len(fingerprints) == len(set(fingerprints))


def test_unknown_license_requires_opt_in(tmp_path) -> None:
    with pytest.raises(ValueError, match="license review"):
        download_public_data(tmp_path, ["ag_news"])


def test_unlabeled_test_row_is_skipped() -> None:
    assert (
        convert_record(SPECS["mnli"], {"premise": "A", "hypothesis": "B", "label": -1}, "vault", 1)
        is None
    )


def test_drive_checkpoint_sync_and_restore(tmp_path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "drive"
    checkpoint = local / "public_baseline" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test checkpoint")
    manifest = local / "dataset_manifest.json"
    manifest.write_text('{"sources":{}}')
    sync = CheckpointSync(local, drive_root=remote)
    assert set(sync.sync_once()) == {"public_baseline/last.pt", "dataset_manifest.json"}
    assert not sync.sync_once()
    checkpoint.unlink()
    manifest.unlink()
    restored = restore_checkpoint("public_baseline/last.pt", local, drive_root=remote)
    assert restored == checkpoint
    assert restored.read_bytes() == b"test checkpoint"
    restored_manifest = restore_checkpoint("dataset_manifest.json", local, drive_root=remote)
    assert restored_manifest == manifest
    assert json.loads(restored_manifest.read_text()) == {"sources": {}}
