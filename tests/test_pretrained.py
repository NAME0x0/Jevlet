from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from jevlet.attention import assert_branch_isolation  # noqa: E402
from jevlet.data import DecisionExample, Question, permute_question  # noqa: E402
from jevlet.pretrained import (  # noqa: E402
    DEFAULT_BACKBONE,
    DEFAULT_CACHE,
    PretrainedConfig,
    PretrainedJevlet,
    build_pretrained_mask,
)
from jevlet.synthetic import generate_dataset  # noqa: E402
from jevlet.training import load_checkpoint, train_experiment  # noqa: E402

CACHED = (DEFAULT_CACHE / ("models--" + DEFAULT_BACKBONE.replace("/", "--"))).exists()
pytestmark = pytest.mark.skipif(not CACHED, reason="pretrained backbone is not cached locally")


@pytest.fixture(scope="module")
def model() -> PretrainedJevlet:
    torch.manual_seed(5)
    return PretrainedJevlet(PretrainedConfig()).eval()


def _example(options_a=("Codex", "Human", "Local"), sibling="Is the task risky?"):
    return DecisionExample(
        "x",
        "A Python unit test fails after a refactor and needs a code fix.",
        [
            Question("Who should handle this?", list(options_a), 0),
            Question(sibling, ["True", "False"], 1, "noul"),
        ],
        "test",
        "test",
        "dev",
    )


def _logits(model: PretrainedJevlet, example: DecisionExample, topology: str):
    with torch.no_grad():
        return model(model.make_collator(topology)([example])).logits


def test_mask_isolates_sibling_branches(model) -> None:
    batch = model.make_collator("block_bidir")([_example()])
    mask = build_pretrained_mask(
        batch["branch_ids"],
        batch["role_ids"],
        batch["option_ids"],
        batch["valid_mask"],
        "block_bidir",
    )
    assert_branch_isolation(mask[:, 0], batch["branch_ids"])
    state = batch["branch_ids"][0] == -1
    assert not mask[0, 0][state][:, ~state].any(), "state tokens must not see question branches"


def test_packed_state_sharing_matches_separate_encoding(model) -> None:
    packed = _logits(model, _example(), "block_bidir")
    separate = _logits(model, _example(), "separate")
    for left, right in zip(packed, separate, strict=True):
        torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)


def test_sibling_question_cannot_change_decision_but_full_topology_leaks(model) -> None:
    base = _logits(model, _example(), "block_bidir")[0]
    edited = _logits(model, _example(sibling="Would a human enjoy this?"), "block_bidir")[0]
    torch.testing.assert_close(base, edited, atol=1e-5, rtol=1e-5)
    leaky_base = _logits(model, _example(), "full")[0]
    leaky_edited = _logits(model, _example(sibling="Would a human enjoy this?"), "full")[0]
    assert not torch.allclose(leaky_base, leaky_edited, atol=1e-5)


def test_option_isolated_topology_is_exactly_order_invariant(model) -> None:
    example = _example()
    permutation = [2, 0, 1]
    question = permute_question(example.questions[0], permutation)
    permuted = DecisionExample("y", example.state, [question], "test", "test", "dev")
    original = DecisionExample("x", example.state, [example.questions[0]], "test", "test", "dev")
    base = _logits(model, original, "option_isolated")[0].softmax(-1)
    changed = _logits(model, permuted, "option_isolated")[0].softmax(-1)
    torch.testing.assert_close(changed, base[permutation], atol=1e-5, rtol=1e-5)
    listwise = _logits(model, original, "block_bidir")[0].softmax(-1)
    listwise_changed = _logits(model, permuted, "block_bidir")[0].softmax(-1)
    assert not torch.allclose(listwise_changed, listwise[permutation], atol=1e-5)


def test_branch_positions_restart_after_state(model) -> None:
    batch = model.make_collator("block_bidir")([_example()])
    positions = batch["position_ids"][0]
    branches = batch["branch_ids"][0]
    state_len = int((branches == -1).sum())
    for branch in (0, 1):
        assert int(positions[branches == branch].min()) == state_len


def test_position_budget_is_enforced() -> None:
    config = PretrainedConfig(max_state_tokens=500, max_question_tokens=64)
    small = PretrainedJevlet(config, load_weights=False)
    example = DecisionExample(
        "long", "word " * 600, [Question("pick " * 40, ["a", "b"], 0)], "t", "t", "dev"
    )
    with pytest.raises(ValueError, match="position"):
        small.make_collator()([example])


def test_pretrained_training_checkpoint_round_trip(tmp_path) -> None:
    data_root = tmp_path / "data"
    generate_dataset(data_root, count=36, seed=3)
    config = {
        "seed": 3,
        "device": "cpu",
        "model": {"family": "pretrained", "max_state_tokens": 48, "option_pool": "mean"},
        "data": {"train": str(data_root / "train.jsonl"), "dev": str(data_root / "dev.jsonl")},
        "training": {
            "batch_size": 2,
            "gradient_accumulation": 1,
            "max_steps": 2,
            "learning_rate": 1e-3,
            "backbone_learning_rate": 2e-5,
            "lr_schedule": "cosine",
            "warmup_steps": 1,
            "amp_dtype": "none",
        },
        "evaluation": {"batch_size": 4, "permutation_examples": 2, "benchmark_repeats": 1},
    }
    metrics = train_experiment(config, tmp_path / "run")
    assert metrics["steps"] == 2
    assert metrics["shared_state_speedup"] is not None
    reloaded, payload = load_checkpoint(tmp_path / "run" / "best.pt")
    example = _example()
    collator = reloaded.make_collator()
    with torch.no_grad():
        logits = reloaded(collator([example])).logits
    assert [tensor.shape[0] for tensor in logits] == [3, 2]
    assert payload["model_config"]["family"] == "pretrained"
