from __future__ import annotations

import json

import pytest
import torch

from jevlet.export import export_checkpoint
from jevlet.feedback import FeedbackStore
from jevlet.personalize import feedback_partitions, personalize, validated_examples
from jevlet.synthetic import generate_dataset
from jevlet.training import train_experiment

OPTIONS = (("Alpha", "Handles red requests"), ("Beta", "Handles blue requests"))


@pytest.fixture()
def setup(tmp_path):
    generate_dataset(tmp_path / "data", count=120, seed=8)
    config = {
        "seed": 8,
        "device": "cpu",
        "model": {
            "max_seq_len": 384,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "ffn_mult": 2,
            "dropout": 0.0,
            "gradient_checkpointing": False,
        },
        "data": {
            "train": str(tmp_path / "data/train.jsonl"),
            "dev": str(tmp_path / "data/dev.jsonl"),
        },
        "training": {"batch_size": 2, "max_steps": 1, "fp16": False},
        "evaluation": {"batch_size": 8, "permutation_examples": 0, "benchmark_repeats": 1},
    }
    train_experiment(config, tmp_path / "base")
    current = export_checkpoint(tmp_path / "base/best.pt", tmp_path / "daily/current.pt")
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    for index in range(60):
        colour = "red" if index % 2 else "blue"
        identifier = store.log_decision(f"Task: a {colour} request", "Who?", OPTIONS, "Alpha")
        if colour == "red":
            store.submit_feedback(identifier, approved=True)
        else:
            store.submit_feedback(identifier, approved=False, corrected_option="Beta")
    return current, store, tmp_path / "data/dev.jsonl"


def test_demonstrations_join_the_partitions(setup, tmp_path) -> None:
    from jevlet.desktop.demonstrations import Demonstration, DemonstrationStore

    _, store, _ = setup
    demos = DemonstrationStore(tmp_path / "demos.sqlite3")
    control = {
        "name": "Reply",
        "automation_id": "",
        "control_type": "Button",
        "rect": [0, 0, 1, 1],
        "enabled": True,
    }
    for _ in range(5):
        demos.record(Demonstration("answer omar", "Inbox", "OUTLOOK", (control,), control))
    parts = feedback_partitions(store, tmp_path / "demos.sqlite3")
    demo_rows = [ex for rows in parts.values() for ex in rows if ex.family == "demonstration"]
    assert len(demo_rows) == 5


def test_partitions_are_stable_and_disjoint(setup) -> None:
    _, store, _ = setup
    parts = feedback_partitions(store)
    ids = [example.example_id for rows in parts.values() for example in rows]
    assert len(ids) == len(set(ids)) == 60
    assert [len(parts[name]) for name in ("train", "tune", "gate")] == [36, 12, 12]


def test_learnable_feedback_is_promoted_with_rollback(setup) -> None:
    current, store, replay = setup
    original = torch.load(current, weights_only=True)["model_state"]
    report = personalize(
        current, store, replay, steps=80, learning_rate=3e-3, max_replay_drop=1.0, device="cpu"
    )
    assert report.promoted, report
    assert report.after["gate_accuracy"] >= report.before["gate_accuracy"]
    assert report.after["gate_nll"] < report.before["gate_nll"]
    previous = torch.load(current.with_name("previous.pt"), weights_only=True)["model_state"]
    for name, value in original.items():
        torch.testing.assert_close(previous[name], value)
    sidecar = json.loads(current.with_suffix(".json").read_text())
    assert sidecar["validated_examples"] == 12 and not sidecar["calibrated"]
    assert validated_examples(current) == 12


def test_failed_gate_leaves_current_model_untouched(setup) -> None:
    current, store, replay = setup
    before = current.read_bytes()
    report = personalize(
        current, store, replay, steps=5, learning_rate=3e-3, max_replay_drop=-1.0, device="cpu"
    )
    assert not report.promoted
    assert current.read_bytes() == before
    assert not current.with_name("previous.pt").exists()
