from __future__ import annotations

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
