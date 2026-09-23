"""Command-line-friendly inference helpers."""

from __future__ import annotations

from .api import JevletRouter


def decide(checkpoint: str, state: str, question: str, options: list[str]) -> dict:
    result = JevletRouter(checkpoint).choice(state, question, options)
    return {
        "selected": result.selected,
        "confidence": result.confidence,
        "probabilities": dict(zip(result.options, result.probabilities, strict=True)),
    }
