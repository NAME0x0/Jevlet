"""Small local decision/router API with Noul, Choice, and Score results."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .data import DecisionCollator, DecisionExample, Question
from .training import load_checkpoint, move_batch


@dataclass(frozen=True, slots=True)
class Choice:
    options: tuple[str, ...]
    probabilities: tuple[float, ...]
    selected: str
    confidence: float


@dataclass(frozen=True, slots=True)
class Noul:
    probability_true: float
    probability_false: float
    probability_unknown: float | None
    value: bool | None
    confidence: float


@dataclass(frozen=True, slots=True)
class Score:
    values: tuple[float, ...]
    probabilities: tuple[float, ...]
    expectation: float
    selected: float
    confidence: float


class JevletRouter:
    def __init__(self, checkpoint: str, device: str = "auto") -> None:
        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved == "auto":
            resolved = "cpu"
        self.device = torch.device(resolved)
        self.model, payload = load_checkpoint(checkpoint, self.device)
        self.temperature = float(payload.get("temperature", 1.0))
        data_config = payload.get("training_config", {}).get("data", {})
        self.collator = DecisionCollator(
            max_seq_len=self.model.config.max_seq_len,
            attention_topology=self.model.config.attention_topology,
            max_state_bytes=int(data_config.get("max_state_bytes", 128)),
            max_question_bytes=int(data_config.get("max_question_bytes", 64)),
            max_option_bytes=int(data_config.get("max_option_bytes", 48)),
        )

    @torch.no_grad()
    def _probabilities(
        self, state: str, question: str, options: Sequence[str]
    ) -> tuple[float, ...]:
        if len(options) < 2:
            raise ValueError("a decision requires at least two options")
        example = DecisionExample(
            "inference",
            state,
            [Question(question, list(options), 0)],
            "inference",
            "local",
            "inference",
        )
        output = self.model(move_batch(self.collator([example]), self.device))
        probabilities = (output.logits[0].float() / self.temperature).softmax(-1).cpu().tolist()
        return tuple(probabilities)

    def choice(self, state: str, question: str, options: Sequence[str]) -> Choice:
        probabilities = self._probabilities(state, question, options)
        selected_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return Choice(
            tuple(options),
            probabilities,
            options[selected_index],
            probabilities[selected_index],
        )

    def noul(self, state: str, question: str, allow_unknown: bool = True) -> Noul:
        options = ["True", "False"] + (["Unknown"] if allow_unknown else [])
        result = self.choice(state, question, options)
        selected_index = result.options.index(result.selected)
        value = True if selected_index == 0 else False if selected_index == 1 else None
        return Noul(
            result.probabilities[0],
            result.probabilities[1],
            result.probabilities[2] if allow_unknown else None,
            value,
            result.confidence,
        )

    def score(
        self,
        state: str,
        question: str,
        values: Sequence[float],
        labels: Sequence[str] | None = None,
    ) -> Score:
        if labels is None:
            labels = [str(value) for value in values]
        if len(values) != len(labels):
            raise ValueError("values and labels must have equal lengths")
        result = self.choice(state, question, labels)
        expectation = sum(
            value * probability
            for value, probability in zip(values, result.probabilities, strict=True)
        )
        selected_index = result.options.index(result.selected)
        return Score(
            tuple(values),
            result.probabilities,
            expectation,
            values[selected_index],
            result.confidence,
        )

    def choice_two_stage(
        self,
        state: str,
        question: str,
        options: Sequence[str],
        independent_scorer: Callable[[str, str, str], float],
        top_k: int = 16,
    ) -> Choice:
        """Cheap independent pruning followed by the ordinary listwise final choice."""

        if top_k < 2:
            raise ValueError("top_k must be at least two")
        shortlisted = sorted(
            options,
            key=lambda option: independent_scorer(state, question, option),
            reverse=True,
        )[:top_k]
        return self.choice(state, question, shortlisted)


def confidence_gate(
    choice: Choice, execute_threshold: float = 0.90, verify_threshold: float = 0.60
) -> str:
    if choice.confidence >= execute_threshold:
        return "execute"
    if choice.confidence >= verify_threshold:
        return "verify"
    return "escalate"
