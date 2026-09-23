from __future__ import annotations

import torch

from jevlet.attention import build_attention_mask
from jevlet.data import DecisionCollator, DecisionExample, Question, permute_question
from jevlet.model import JevletModel, ModelConfig


def _model() -> JevletModel:
    torch.manual_seed(11)
    return JevletModel(
        ModelConfig(
            max_seq_len=192,
            d_model=32,
            n_layers=2,
            n_heads=4,
            ffn_mult=2,
            dropout=0.0,
            attention_topology="block_causal",
            gradient_checkpointing=False,
        )
    ).eval()


def test_later_option_can_use_earlier_option_context() -> None:
    collator = DecisionCollator(max_seq_len=192, attention_topology="block_causal")
    question_a = Question("Choose.", ["red", "blue"], 0)
    question_b = Question("Choose.", ["cat", "blue"], 0)
    example_a = DecisionExample("a", "state", [question_a], "test", "test", "dev")
    example_b = DecisionExample("b", "state", [question_b], "test", "test", "dev")
    batch_a = collator([example_a])
    batch_b = collator([example_b])
    second_end = batch_a["records"][0]["option_end_positions"][1]
    first_start = batch_a["records"][0]["option_spans"][0][0]
    allowed = build_attention_mask(batch_a["branch_ids"], batch_a["valid_mask"])
    assert allowed[0, second_end, first_start]
    hidden_a = _model()(batch_a, return_hidden=True).hidden_states
    hidden_b = _model()(batch_b, return_hidden=True).hidden_states
    assert hidden_a is not None and hidden_b is not None
    assert not torch.equal(hidden_a[0, second_end], hidden_b[0, second_end])


def test_dynamic_option_count_and_permutation_label_mapping() -> None:
    model = _model()
    collator = DecisionCollator(max_seq_len=192)
    original = Question("Route.", ["local", "Codex", "human"], 1, target_probs=[0.1, 0.8, 0.1])
    changed = permute_question(original, [2, 0, 1])
    assert changed.label == 2
    assert changed.target_probs == [0.1, 0.1, 0.8]
    example = DecisionExample("x", "task", [changed], "test", "test", "dev")
    output = model(collator([example]))
    assert output.logits[0].shape == (3,)


def test_separate_topology_duplicates_state_per_question() -> None:
    collator = DecisionCollator(max_seq_len=192, attention_topology="separate")
    questions = [Question("Q1", ["a", "b"], 0), Question("Q2", ["c", "d"], 1)]
    example = DecisionExample("x", "shared", questions, "test", "test", "dev")
    batch = collator([example])
    assert batch["token_ids"].shape[0] == 2
    assert [record["row"] for record in batch["records"]] == [0, 1]
