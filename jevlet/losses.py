"""Proper scoring rules for categorical decisions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def target_distribution(logits: torch.Tensor, record: dict[str, Any]) -> torch.Tensor:
    if record.get("target_probs") is not None:
        target = logits.new_tensor(record["target_probs"])
        return target / target.sum().clamp_min(1e-12)
    return F.one_hot(
        torch.tensor(record["label"], device=logits.device), num_classes=logits.numel()
    ).to(logits.dtype)


def decision_loss(
    logits_list: list[torch.Tensor],
    records: list[dict[str, Any]],
    loss_name: str = "ce",
    brier_weight: float = 0.25,
    label_smoothing: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(logits_list) != len(records) or not logits_list:
        raise ValueError("loss requires one non-empty logit vector per record")

    cross_entropies = []
    brier_scores = []
    for logits, record in zip(logits_list, records, strict=True):
        target = target_distribution(logits, record)
        if loss_name == "label_smooth_ce" and record.get("target_probs") is None:
            target = target * (1.0 - label_smoothing) + label_smoothing / target.numel()
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        cross_entropies.append(-(target * log_probabilities).sum())
        brier_scores.append(((probabilities - target) ** 2).sum())

    ce = torch.stack(cross_entropies).mean()
    brier = torch.stack(brier_scores).mean()
    if loss_name in {"ce", "label_smooth_ce"}:
        total = ce
    elif loss_name == "brier":
        total = brier
    elif loss_name == "ce_brier":
        total = ce + brier_weight * brier
    else:
        raise ValueError(f"unknown loss: {loss_name}")
    return total, {"ce": float(ce.detach()), "brier": float(brier.detach())}
