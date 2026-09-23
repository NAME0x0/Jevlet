from __future__ import annotations

import hashlib

from jevlet.synthetic import generate_dataset


def test_dataset_generation_is_deterministic_and_separates_vault(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest_a = generate_dataset(first, count=120, seed=42)
    manifest_b = generate_dataset(second, count=120, seed=42)
    assert manifest_a["splits"]["train"]["sha256"] == manifest_b["splits"]["train"]["sha256"]
    assert (first / "vault" / "vault.jsonl").exists()
    assert not (first / "vault.jsonl").exists()
    digest = hashlib.sha256((first / "dev.jsonl").read_bytes()).hexdigest()
    assert digest == manifest_a["splits"]["dev"]["sha256"]
