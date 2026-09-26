from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from jevlet.synthetic import generate_dataset
from jevlet.training import train_experiment


def _config(data_root):
    return {
        "seed": 17,
        "device": "cpu",
        "model": {
            "max_seq_len": 256,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "ffn_mult": 2,
            "dropout": 0.1,
            "attention_topology": "block_causal",
            "option_pool": "end",
            "state_pool": "last",
            "decision_head": "pointer",
            "gradient_checkpointing": False,
        },
        "data": {
            "train": str(data_root / "train.jsonl"),
            "dev": str(data_root / "dev.jsonl"),
            "max_state_bytes": 64,
            "max_question_bytes": 48,
            "max_option_bytes": 24,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation": 1,
            "max_steps": 1,
            "save_every_steps": 1,
            "fp16": False,
            "loss": "ce",
        },
        "evaluation": {
            "batch_size": 4,
            "permutation_examples": 0,
            "benchmark_repeats": 1,
        },
    }


def test_resumed_training_matches_uninterrupted_weights(tmp_path) -> None:
    data_root = tmp_path / "data"
    generate_dataset(data_root, count=48, seed=17)
    config = _config(data_root)
    first_dir = tmp_path / "first"
    train_experiment(config, first_dir)

    resumed = deepcopy(config)
    resumed["training"]["max_steps"] = 2
    resumed["training"]["resume_from"] = str(first_dir / "last.pt")
    resumed_dir = tmp_path / "resumed"
    train_experiment(resumed, resumed_dir)

    full = deepcopy(config)
    full["training"]["max_steps"] = 2
    full_dir = tmp_path / "full"
    train_experiment(full, full_dir)

    resumed_state = torch.load(resumed_dir / "last.pt", weights_only=True)
    full_state = torch.load(full_dir / "last.pt", weights_only=True)
    assert resumed_state["step"] == full_state["step"] == 2
    for name, value in full_state["model_state"].items():
        torch.testing.assert_close(resumed_state["model_state"][name], value)


def test_resume_on_a_smaller_micro_batch_continues_at_the_same_row(tmp_path) -> None:
    # A new Colab session can land on a smaller GPU: batch 2 x 2 replaces batch 4 x 1.
    data_root = tmp_path / "data"
    generate_dataset(data_root, count=48, seed=17)
    config = _config(data_root)
    config["training"].update(batch_size=4, max_steps=2)
    first_dir = tmp_path / "first"
    train_experiment(config, first_dir)
    assert torch.load(first_dir / "last.pt", weights_only=True)["example_offset"] == 8

    resumed = deepcopy(config)
    resumed["training"].update(
        batch_size=2, gradient_accumulation=2, max_steps=3, resume_from=str(first_dir / "last.pt")
    )
    train_experiment(resumed, tmp_path / "resumed")
    state = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    assert state["step"] == 3 and state["example_offset"] == 12


def test_run_files_are_mirrored_and_snapshots_are_weights_only(tmp_path) -> None:
    data_root = tmp_path / "data"
    generate_dataset(data_root, count=48, seed=17)
    config = _config(data_root)
    mirror = tmp_path / "drive" / "run"
    config["training"].update(
        max_steps=3, keep_steps=[2], log_every_steps=1, mirror_dir=str(mirror)
    )
    train_experiment(config, tmp_path / "run")
    for name in (
        "last.pt",
        "best.pt",
        "metrics.json",
        "config.json",
        "progress.jsonl",
        "step-2.pt",
    ):
        assert (mirror / name).exists(), name
    assert not list(mirror.glob("*.tmp"))
    snapshot = torch.load(mirror / "step-2.pt", weights_only=True)
    assert "optimizer_state" not in snapshot and snapshot["metrics"]["steps"] == 2
    assert len((mirror / "progress.jsonl").read_text().splitlines()) == 3


def test_resume_rejects_changed_training_data(tmp_path) -> None:
    data_root = tmp_path / "data"
    generate_dataset(data_root, count=48, seed=17)
    config = _config(data_root)
    first_dir = tmp_path / "first"
    train_experiment(config, first_dir)
    first_line = (data_root / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    with (data_root / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(first_line + "\n")
    resumed = deepcopy(config)
    resumed["training"]["max_steps"] = 2
    resumed["training"]["resume_from"] = str(first_dir / "last.pt")
    with pytest.raises(ValueError, match="training data differs"):
        train_experiment(resumed, tmp_path / "resumed")


def test_resume_keeps_the_earlier_progress_log_and_totals_the_run(tmp_path) -> None:
    # A new Colab VM resumes into an empty run directory: the loss log of the first session
    # must come back from Drive, and time must add up across sessions.
    data_root = tmp_path / "data"
    generate_dataset(data_root, count=48, seed=17)
    config = _config(data_root)
    mirror = tmp_path / "drive" / "run"
    config["training"].update(max_steps=2, log_every_steps=1, mirror_dir=str(mirror))
    train_experiment(config, tmp_path / "first")
    first = torch.load(mirror / "last.pt", weights_only=True)["run_totals"]
    assert first["sessions"] == 1 and first["complete"] and first["train_seconds"] > 0
    # A row logged after the last checkpoint (then lost in a crash) and a half-written line.
    with (mirror / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"step": 3, "loss": 1.0}\n{"step": 4, "lo')

    resumed = deepcopy(config)
    resumed["training"].update(max_steps=4, resume_from=str(mirror / "last.pt"))
    metrics = train_experiment(resumed, tmp_path / "second")
    steps = [
        json.loads(line)["step"] for line in (mirror / "progress.jsonl").read_text().splitlines()
    ]
    assert steps == [1, 2, 3, 4]
    assert metrics["training_sessions"] == 2 and metrics["train_seconds_complete"]
    assert metrics["train_seconds"] > metrics["train_seconds_this_session"]


def test_resume_from_a_checkpoint_without_totals_marks_time_incomplete(tmp_path) -> None:
    data_root = tmp_path / "data"
    generate_dataset(data_root, count=48, seed=17)
    config = _config(data_root)
    train_experiment(config, tmp_path / "first")
    old = torch.load(tmp_path / "first" / "last.pt", weights_only=True)
    del old["run_totals"]  # as written before run totals existed (v6's first session)
    torch.save(old, tmp_path / "old.pt")
    resumed = deepcopy(config)
    resumed["training"].update(max_steps=2, resume_from=str(tmp_path / "old.pt"))
    metrics = train_experiment(resumed, tmp_path / "second")
    assert metrics["train_seconds_complete"] is False
