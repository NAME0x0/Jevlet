from __future__ import annotations

import json

from jevlet.data import JsonlDecisionDataset
from jevlet.mixture import build_mixture
from jevlet.synthetic import generate_dataset


def test_mixture_caps_sources_and_keeps_vault_separate(tmp_path) -> None:
    generate_dataset(tmp_path / "a", count=60, seed=1)
    generate_dataset(tmp_path / "b", count=40, seed=2)
    manifest = build_mixture(
        {"a": tmp_path / "a", "b": tmp_path / "b"},
        tmp_path / "mix",
        limits={"a": {"train": 10}},
        seed=7,
    )
    assert manifest["sources"]["a"]["splits"]["train"]["rows"] == 10
    assert manifest["splits"]["train"]["rows"] == 10 + 32
    assert (tmp_path / "mix" / "vault" / "vault.jsonl").exists()
    assert not (tmp_path / "mix" / "vault.jsonl").exists()
    ids = {example.example_id for example in JsonlDecisionDataset(tmp_path / "mix" / "train.jsonl")}
    vault = JsonlDecisionDataset(tmp_path / "mix" / "vault" / "vault.jsonl")
    assert not ids & {example.example_id for example in vault}


def test_mixture_is_deterministic(tmp_path) -> None:
    generate_dataset(tmp_path / "a", count=60, seed=1)
    first = build_mixture({"a": tmp_path / "a"}, tmp_path / "one", limits={"a": {"train": 9}})
    second = build_mixture({"a": tmp_path / "a"}, tmp_path / "two", limits={"a": {"train": 9}})
    assert json.dumps(first["splits"]["train"]["sha256"]) == json.dumps(
        second["splits"]["train"]["sha256"]
    )
