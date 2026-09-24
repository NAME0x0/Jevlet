"""Post-hoc temperature scaling on a held-out development split."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .losses import target_distribution


def fit_temperature(
    logits_list: list[torch.Tensor], records: list[dict[str, Any]], max_iter: int = 50
) -> float:
    if not logits_list:
        return 1.0
    grouped: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for logits, record in zip(logits_list, records, strict=True):
        detached = logits.detach().float()
        logits_group, target_group = grouped.setdefault(detached.numel(), ([], []))
        logits_group.append(detached)
        target_group.append(target_distribution(detached, record))
    batches = [
        (torch.stack(logits_group), torch.stack(target_group))
        for logits_group, target_group in grouped.values()
    ]
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        losses = []
        weights = []
        for logits, target in batches:
            scaled = logits / temperature
            losses.append(-(target * F.log_softmax(scaled, dim=-1)).sum(dim=-1).mean())
            weights.append(logits.shape[0])
        loss = sum(item * weight for item, weight in zip(losses, weights, strict=True)) / sum(
            weights
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def fit_temperatures_by_kind(
    logits_list: list[torch.Tensor], records: list[dict[str, Any]], minimum: int = 20
) -> dict[str, float]:
    """One temperature per question kind (choice/noul/score), known at inference time.

    A single global temperature is dominated by whichever family fills the calibration set;
    yes/no and many-option questions need different corrections. Kinds with fewer than
    ``minimum`` rows fall back to the pooled temperature.
    """
    pooled = fit_temperature(logits_list, records)
    temperatures = {"default": pooled}
    kinds = sorted({str(record.get("kind", "choice")) for record in records})
    for kind in kinds:
        pairs = [
            (logits, record)
            for logits, record in zip(logits_list, records, strict=True)
            if str(record.get("kind", "choice")) == kind
        ]
        if len(pairs) >= minimum:
            temperatures[kind] = fit_temperature([p[0] for p in pairs], [p[1] for p in pairs])
    return temperatures
