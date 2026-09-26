"""Decision quality and calibration metrics."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .losses import target_distribution


def expected_calibration_error(
    confidences: torch.Tensor, correct: torch.Tensor, bins: int = 15
) -> float:
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    error = torch.tensor(0.0)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        members = (confidences > lower) & (confidences <= upper)
        if members.any():
            accuracy = correct[members].float().mean()
            confidence = confidences[members].mean()
            error += members.float().mean() * (accuracy - confidence).abs()
    return float(error)


def target_calibration_error(
    confidences: torch.Tensor, target_mass: torch.Tensor, bins: int = 15
) -> float:
    """ECE against the target distribution instead of hard correctness.

    ``target_mass`` is the target probability of the predicted option. With one-hot targets it
    equals correctness and this is the ordinary ECE; with soft targets (the risk question's
    ``[risk, 1 - risk]``) it asks whether a 0.96 confidence meets a 0.96 target, which hard ECE
    would score as a 0.04 miscalibration even when the model reproduces its target exactly.
    """
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    error = torch.tensor(0.0)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        members = (confidences > lower) & (confidences <= upper)
        if members.any():
            gap = target_mass[members].mean() - confidences[members].mean()
            error += members.float().mean() * gap.abs()
    return float(error)


def compute_metrics(
    logits_list: list[torch.Tensor], records: list[dict[str, Any]], temperature: float = 1.0
) -> dict[str, float | int | None]:
    if not logits_list:
        return {"questions": 0}
    correct_values = []
    confidences = []
    target_masses = []
    target_distances = []
    nll_values = []
    brier_values = []
    unknown_correct = []
    held_out_correct = []
    soft = 0
    for logits, record in zip(logits_list, records, strict=True):
        scaled = logits.float() / temperature
        probabilities = F.softmax(scaled, dim=-1)
        prediction = int(probabilities.argmax())
        correct = prediction == record["label"]
        target = target_distribution(scaled, record)
        soft += record.get("target_probs") is not None
        correct_values.append(correct)
        confidences.append(float(probabilities.max()))
        target_masses.append(float(target[prediction]))
        target_distances.append(float(0.5 * (probabilities - target).abs().sum()))
        nll_values.append(float(-(target * probabilities.clamp_min(1e-12).log()).sum()))
        brier_values.append(float(((probabilities - target) ** 2).sum()))
        if record.get("is_unknown"):
            unknown_correct.append(correct)
        if record.get("is_ood"):
            held_out_correct.append(correct)
    correct_tensor = torch.tensor(correct_values, dtype=torch.bool)
    confidence_tensor = torch.tensor(confidences)
    return {
        "questions": len(records),
        "accuracy": sum(correct_values) / len(correct_values),
        # Examples whose domain (for grounding: the app) never appears in training. Only the
        # grounding generator sets this, so it is not a general out-of-distribution score.
        "held_out_domain_accuracy": (
            sum(held_out_correct) / len(held_out_correct) if held_out_correct else None
        ),
        "unknown_accuracy": (
            sum(unknown_correct) / len(unknown_correct) if unknown_correct else None
        ),
        "brier": sum(brier_values) / len(brier_values),
        # Hard-label ECE: confidence against argmax correctness. Misleading on soft-target
        # questions; read target_ece there.
        "ece": expected_calibration_error(confidence_tensor, correct_tensor),
        "target_ece": target_calibration_error(confidence_tensor, torch.tensor(target_masses)),
        # Mean total-variation distance between the predicted and target distributions.
        "target_distance": sum(target_distances) / len(target_distances),
        "soft_target_fraction": soft / len(records),
        "nll": sum(nll_values) / len(nll_values),
        "mean_confidence": sum(confidences) / len(confidences),
    }


def metrics_by_family(
    logits_list: list[torch.Tensor], records: list[dict[str, Any]], temperature: float = 1.0
) -> dict[str, dict[str, float | int | None]]:
    """Per-family metrics, so an aggregate cannot hide a collapsed source."""
    groups: dict[str, tuple[list[torch.Tensor], list[dict[str, Any]]]] = {}
    for logits, record in zip(logits_list, records, strict=True):
        bucket = groups.setdefault(str(record.get("family", "unknown")), ([], []))
        bucket[0].append(logits)
        bucket[1].append(record)
    return {
        family: compute_metrics(family_logits, family_records, temperature)
        for family, (family_logits, family_records) in sorted(groups.items())
    }


def finite_metrics(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True
