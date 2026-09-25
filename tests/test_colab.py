"""The Colab driver's crash handling, exercised locally with a tiny model and fake Drive."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from jevlet import colab
from jevlet.colab import CACHE_FILES, Paths, fit_batch
from jevlet.model_card import render_model_card
from jevlet.synthetic import generate_dataset

REPO = Path(__file__).resolve().parents[1]


def _paths(tmp_path: Path) -> Paths:
    return Paths(drive=tmp_path / "drive", work=tmp_path / "vm", repo=REPO, version="test")


def _tiny_config(tmp_path: Path) -> Path:
    config = {
        "seed": 5,
        "device": "cpu",
        "model": {
            "max_seq_len": 256, "d_model": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2,
            "dropout": 0.0, "attention_topology": "block_causal", "option_pool": "end",
            "state_pool": "last", "decision_head": "pointer", "gradient_checkpointing": False,
        },
        "data": {"max_state_bytes": 64, "max_question_bytes": 48, "max_option_bytes": 24},
        "training": {
            "effective_batch": 4, "batch_size": 4, "gradient_accumulation": 1, "max_steps": 4,
            "save_every_steps": 2, "keep_steps": [2], "log_every_steps": 1, "fp16": False,
            "loss": "ce",
        },
        "evaluation": {"batch_size": 4, "permutation_examples": 0, "benchmark_repeats": 1},
    }  # fmt: skip
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(config))
    return path


def _mixture(paths: Paths) -> None:
    generate_dataset(paths.data / "mixture_v6", count=60, seed=5)


def test_fit_batch_keeps_the_effective_batch() -> None:
    assert fit_batch(64, 64) == (64, 1)
    assert fit_batch(64, 32) == (32, 2)
    assert fit_batch(64, 24) == (16, 4)
    assert fit_batch(64, 7) == (4, 16)


def test_data_cache_round_trip_and_tamper_detection(tmp_path) -> None:
    paths = _paths(tmp_path)
    for index, name in enumerate(CACHE_FILES):
        (paths.data / name).parent.mkdir(parents=True, exist_ok=True)
        (paths.data / name).write_text(f"row {index}\n" * 50)
    colab.save_data_cache(paths)
    for name in CACHE_FILES:
        (paths.data / name).unlink()
    assert colab.restore_data_cache(paths)
    assert (paths.data / CACHE_FILES[3]).read_text().startswith("row 3")
    index_path = paths.cache / f"data_{paths.version}.index.json"
    index = json.loads(index_path.read_text())
    index[CACHE_FILES[0]] = "0" * 64
    index_path.write_text(json.dumps(index))
    assert not colab.restore_data_cache(paths)


def test_resume_prefers_a_readable_checkpoint_and_rejects_other_data(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.mirror.mkdir(parents=True)
    torch.save({"train_data_sha256": "abc", "step": 7}, paths.mirror / "last.prev.pt")
    (paths.mirror / "last.pt").write_bytes(b"truncated by a crash")
    resume = colab.latest_resume(paths, "abc")
    assert resume is not None and torch.load(resume, weights_only=True)["step"] == 7
    with pytest.raises(RuntimeError, match="different data"):
        colab.latest_resume(paths, "other")


def test_train_runs_detached_mirrors_to_drive_and_resumes(tmp_path) -> None:
    paths = _paths(tmp_path)
    _mixture(paths)
    base = _tiny_config(tmp_path)
    profile = {"name": "cpu", "memory_gb": 0, "bf16": False, "batch_size": 2, "workers": 0}
    config_path = colab.run_config(paths, profile, str(base))
    config = json.loads(config_path.read_text())
    assert (
        config["training"]["batch_size"] == 2 and config["training"]["gradient_accumulation"] == 2
    )
    assert "resume_from" not in config["training"]
    assert colab.train(paths, profile, str(base)) == "done"
    for name in ("last.pt", "last.prev.pt", "best.pt", "metrics.json", "step-2.pt", "train.log"):
        assert (paths.mirror / name).exists(), name
    assert colab.train(paths, profile, str(base)) == "done"  # finished runs are not retrained
    # A reset VM with a longer schedule resumes from Drive instead of starting over.
    (paths.mirror / "metrics.json").unlink()
    stored = json.loads((paths.mirror / "run_config.json").read_text())
    stored["training"]["max_steps"] = 6
    (paths.mirror / "run_config.json").write_text(json.dumps(stored))
    resumed = json.loads(colab.run_config(paths, profile, str(base)).read_text())
    assert resumed["training"]["resume_from"].endswith("resume.pt")
    assert len(resumed["sessions"]) == 3


def test_model_card_renders_every_section() -> None:
    block = {"questions": 100, "accuracy": 0.9, "ece": 0.03, "calibrated_ece": 0.02}
    results = {
        "final": {
            "real_vault": {"examples": 50, "assistant/skill": block, "assistant/slot": block},
            "mixture_dev": {"all": block},
        },
        "scaling": {"step-2500": {"assistant/skill": block}},
        "temperatures": {"choice": 1.1, "noul": 0.9},
        "training": {
            "metrics": {"steps": 25000, "parameter_count": 33_000_000, "train_seconds": 7200},
            "config": {
                "training": {"effective_batch": 64, "learning_rate": 0.002, "amp_dtype": "bf16"},
                "sessions": [{"gpu": {"name": "NVIDIA L4"}}],
                "provenance": {"git_commit": "abcdef1234567890"},
            },
            "data_manifest": {"sources": {"real": {"splits": {"train": {"rows": 182_674}}}}},
        },
    }
    card = render_model_card(
        results, repo_id="u/Jevlet", version="v6", github="https://github.com/u/Jevlet"
    )
    assert card.startswith("---\nlanguage: en\nlicense: other")
    for text in (
        "90.0%",
        "160,000",
        "NVIDIA L4",
        "182,674",
        "abcdef123456",
        "u/Jevlet",
        "CC BY-SA 4.0",
    ):
        assert text in card, text
