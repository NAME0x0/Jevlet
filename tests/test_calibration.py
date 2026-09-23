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
