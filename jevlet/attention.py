"""Attention topology utilities for shared-state, isolated question branches."""

from __future__ import annotations

import torch

STATE_BRANCH = -1
PAD_BRANCH = -2


def build_attention_mask(
    branch_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    topology: str = "block_causal",
) -> torch.Tensor:
    """Return an allowed-attention mask shaped ``[batch, query, key]``.

    Topologies:
      - ``separate``: ordinary causal attention (the collator creates one row per question).
      - ``shared_prefix``: packed causal attention; later siblings can see earlier siblings.
      - ``block_causal``: state is shared, but each question sees only state and its own past.

    Padding queries attend only to themselves to prevent all-masked softmax rows; their output is
    discarded by the model immediately afterward.
    """

    if branch_ids.ndim != 2 or valid_mask.shape != branch_ids.shape:
        raise ValueError("branch_ids and valid_mask must both have shape [batch, sequence]")
    if topology not in {"separate", "shared_prefix", "block_causal"}:
        raise ValueError(f"unknown attention topology: {topology}")

    batch, seq_len = branch_ids.shape
    device = branch_ids.device
    positions = torch.arange(seq_len, device=device)
    causal = positions[None, :, None] >= positions[None, None, :]
    valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]

    if topology in {"separate", "shared_prefix"}:
        allowed = causal & valid_pairs
    else:
        query_branch = branch_ids[:, :, None]
        key_branch = branch_ids[:, None, :]
        query_is_state = query_branch == STATE_BRANCH
        key_is_state = key_branch == STATE_BRANCH
        same_question = (query_branch == key_branch) & (query_branch >= 0)
        topology_allowed = (query_is_state & key_is_state) | (
            ~query_is_state & (key_is_state | same_question)
        )
        allowed = causal & valid_pairs & topology_allowed

    padding_queries = ~valid_mask
    eye = torch.eye(seq_len, dtype=torch.bool, device=device).expand(batch, -1, -1)
    return allowed | (padding_queries[:, :, None] & eye)


def assert_branch_isolation(mask: torch.Tensor, branch_ids: torch.Tensor) -> None:
    """Raise if a non-state branch can attend to a sibling branch."""

    for batch_index in range(mask.shape[0]):
        ids = branch_ids[batch_index]
        for query in range(mask.shape[1]):
            query_branch = int(ids[query])
            if query_branch < 0:
                continue
            visible = ids[mask[batch_index, query]]
            leaked = (visible >= 0) & (visible != query_branch)
            if bool(leaked.any()):
                raise AssertionError(f"question branch {query_branch} can see a sibling")
