from __future__ import annotations

import torch

from jevlet.attention import assert_branch_isolation, build_attention_mask
from jevlet.data import DecisionCollator, DecisionExample, Question
from jevlet.model import JevletModel, ModelConfig


def test_block_causal_mask_isolates_sibling_questions() -> None:
    branch_ids = torch.tensor([[-1, -1, 0, 0, 1, 1]])
    valid = torch.ones_like(branch_ids, dtype=torch.bool)
    allowed = build_attention_mask(branch_ids, valid, "block_causal")
    assert_branch_isolation(allowed, branch_ids)
    assert allowed[0, 5, 0]
    assert allowed[0, 5, 4]
    assert not allowed[0, 5, 2]
    assert not allowed[0, 1, 2]


def test_question_logits_are_invariant_to_sibling_content() -> None:
    torch.manual_seed(7)
    config = ModelConfig(
        max_seq_len=192,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ffn_mult=2,
        dropout=0.0,
        attention_topology="block_causal",
        gradient_checkpointing=False,
    )
    model = JevletModel(config).eval()
    collator = DecisionCollator(max_seq_len=192, attention_topology="block_causal")
    stable = Question("Choose the route.", ["local", "human"], 0)
    first_a = Question("Pick a color.", ["red", "blue"], 0)
    first_b = Question("Pick a color.", ["cat", "wolf"], 0)
    example_a = DecisionExample("a", "shared state", [first_a, stable], "test", "test", "dev")
    example_b = DecisionExample("b", "shared state", [first_b, stable], "test", "test", "dev")
    logits_a = model(collator([example_a])).logits[1]
    logits_b = model(collator([example_b])).logits[1]
    torch.testing.assert_close(logits_a, logits_b, rtol=0.0, atol=0.0)


def test_naive_shared_prefix_exposes_earlier_sibling() -> None:
    branch_ids = torch.tensor([[-1, -1, 0, 0, 1, 1]])
    valid = torch.ones_like(branch_ids, dtype=torch.bool)
    allowed = build_attention_mask(branch_ids, valid, "shared_prefix")
    assert allowed[0, 5, 2]
