from __future__ import annotations

import torch

from jevlet.calibration import fit_temperature
from jevlet.metrics import compute_metrics


def test_temperature_scaling_handles_variable_option_counts_and_does_not_worsen_nll() -> None:
    logits = [
        torch.tensor([5.0, 0.0]),
        torch.tensor([5.0, 0.0]),
        torch.tensor([3.0, 1.0, -1.0]),
    ]
    records = [
        {"label": 0, "target_probs": None},
        {"label": 1, "target_probs": None},
        {"label": 1, "target_probs": [0.1, 0.8, 0.1]},
    ]
    raw = compute_metrics(logits, records)
    temperature = fit_temperature(logits, records)
    calibrated = compute_metrics(logits, records, temperature)
    assert 0.05 <= temperature <= 20.0
    assert calibrated["nll"] <= raw["nll"] + 1e-6


def test_soft_target_questions_are_scored_against_their_targets() -> None:
    # A risk question whose model reproduces its soft target [0.04, 0.96] exactly: hard ECE
    # calls that a 0.04 miscalibration (always "correct" at 0.96 confidence); target ECE is 0.
    target = [0.04, 0.96]
    logits = [torch.log(torch.tensor(target))] * 50
    records = [{"label": 1, "target_probs": target}] * 50
    metrics = compute_metrics(logits, records)
    assert abs(metrics["ece"] - 0.04) < 1e-4
    assert metrics["target_ece"] < 1e-4 and metrics["target_distance"] < 1e-4
    assert metrics["soft_target_fraction"] == 1.0


def test_target_ece_equals_ece_for_hard_labels() -> None:
    logits = [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([3.0, 0.0])]
    records = [{"label": 0, "target_probs": None}, {"label": 0, "target_probs": None}] + [
        {"label": 1, "target_probs": None}
    ]
    metrics = compute_metrics(logits, records)
    assert abs(metrics["ece"] - metrics["target_ece"]) < 1e-6
    assert metrics["soft_target_fraction"] == 0.0


def test_question_roles_split_grounding_control_from_risk() -> None:
    from jevlet.benchmarks import RISK_QUESTION
    from jevlet.colab import question_role

    assert question_role("grounding", 0, "Which on-screen control?") == "control"
    assert question_role("grounding", 1, RISK_QUESTION) == "risk"
    assert question_role("assistant", 0, "Which action?") == "skill"
    assert question_role("assistant", 1, RISK_QUESTION) == "risk"
    assert question_role("assistant", 2, "Which app?") == "slot"
    assert question_role("public_mnli", 0, "Does it follow?") == "decision"
