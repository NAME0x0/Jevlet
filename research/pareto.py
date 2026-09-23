"""Pareto-frontier utilities that preserve metric trade-offs."""

from __future__ import annotations

from typing import Any

MAXIMIZE = (
    "accuracy",
    "ood_accuracy",
    "unknown_accuracy",
    "option_order_prediction_agreement",
    "questions_per_second",
)
MINIMIZE = ("brier", "ece", "nll", "latency_ms_per_question", "peak_vram_mb")


def _value(result: dict[str, Any], metric: str, maximize: bool) -> float:
    value = result.get("metrics", result).get(metric)
    if value is None:
        return float("-inf") if maximize else float("inf")
    return float(value)


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = True
    strictly_better = False
    for metric in MAXIMIZE:
        a, b = _value(left, metric, True), _value(right, metric, True)
        no_worse &= a >= b
        strictly_better |= a > b
    for metric in MINIMIZE:
        a, b = _value(left, metric, False), _value(right, metric, False)
        no_worse &= a <= b
        strictly_better |= a < b
    return bool(no_worse and strictly_better)


def pareto_frontier(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in results
        if not any(dominates(other, candidate) for other in results if other is not candidate)
    ]


def promotion_order(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Peel Pareto layers; use a conservative tie-break only inside each layer."""

    remaining = list(results)
    ordered: list[dict[str, Any]] = []
    while remaining:
        layer = pareto_frontier(remaining)
        layer.sort(
            key=lambda result: (
                result["metrics"].get("accuracy") or 0.0,
                -(result["metrics"].get("calibrated_ece") or 1.0),
                result["metrics"].get("questions_per_second") or 0.0,
            ),
            reverse=True,
        )
        ordered.extend(layer)
        layer_ids = {id(item) for item in layer}
        remaining = [item for item in remaining if id(item) not in layer_ids]
    return ordered
