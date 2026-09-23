"""Jev-shaped local API: one state, many typed questions, one packed forward pass.

Mirrors the public shape of TypeSafe's ``system_one(state, questions)`` call. Every question
branch attends to the shared state and to itself only, so the answers equal those of
separate calls while the state is encoded once. Criteria are runtime-defined
``name -> description`` pairs; results can only ever name one of the supplied options.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Union

import torch

from .api import Choice, Noul, Score
from .data import DecisionExample, Question
from .families import build_collator
from .training import load_checkpoint, move_batch, resolve_device

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10


@dataclass(frozen=True, slots=True)
class ChoiceQuestion:
    instructions: str
    criteria: Mapping[str, str] | Sequence[str]


@dataclass(frozen=True, slots=True)
class NoulQuestion:
    instructions: str
    allow_unknown: bool = False


@dataclass(frozen=True, slots=True)
class ScoreQuestion:
    instructions: str
    levels: Sequence[str]


TypedQuestion = Union[ChoiceQuestion, NoulQuestion, ScoreQuestion]  # noqa: UP007
Answer = Union[Choice, Noul, Score]  # noqa: UP007


def option_text(name: str, description: str) -> str:
    """The text a model reads for one option; the name alone when no description adds."""
    name, description = name.strip(), description.strip()
    return name if not description or description == name else f"{name}: {description}"


def _criteria(question: ChoiceQuestion) -> tuple[tuple[str, ...], list[str]]:
    if isinstance(question.criteria, Mapping):
        pairs = [(str(name), str(text)) for name, text in question.criteria.items()]
    else:
        pairs = [(str(name), str(name)) for name in question.criteria]
    names = tuple(name for name, _ in pairs)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("a choice needs at least two uniquely named options")
    if len(names) > MAX_CHOICE_OPTIONS:
        raise ValueError(
            f"a choice supports at most {MAX_CHOICE_OPTIONS} options; shortlist first "
            "(score independently, then choose among the survivors)"
        )
    return names, [option_text(name, text) for name, text in pairs]


class SystemOne:
    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        *,
        execute_threshold: float = 0.90,
        verify_threshold: float = 0.65,
        calibrated: bool = False,
    ) -> None:
        if not 0 < verify_threshold <= execute_threshold <= 1:
            raise ValueError("confidence thresholds must satisfy 0 < verify <= execute <= 1")
        self.device = resolve_device(device)
        self.checkpoint_path = str(checkpoint)
        self.model, payload = load_checkpoint(checkpoint, self.device)
        self.temperature = float(payload.get("temperature", 1.0))
        self.collator = build_collator(self.model, payload.get("training_config", {}).get("data"))
        self.model_id = str(payload.get("model_config", {}).get("backbone", "jevlet-scratch"))
        self.execute_threshold = execute_threshold
        self.verify_threshold = verify_threshold
        # Only feedback validated on the user's own decisions may unlock unattended execution.
        self.calibrated = calibrated

    @torch.no_grad()
    def evaluate(self, state: str, questions: Mapping[str, TypedQuestion]) -> dict[str, Answer]:
        if not questions:
            raise ValueError("ask at least one question")
        specs: list[tuple[str, TypedQuestion, tuple[str, ...]]] = []
        packed: list[Question] = []
        for identifier, question in questions.items():
            if isinstance(question, ChoiceQuestion):
                names, texts = _criteria(question)
                packed.append(Question(question.instructions, texts, 0, "choice"))
            elif isinstance(question, NoulQuestion):
                names = ("True", "False") + (("Unknown",) if question.allow_unknown else ())
                packed.append(Question(question.instructions, list(names), 0, "noul"))
            elif isinstance(question, ScoreQuestion):
                names = tuple(str(level) for level in question.levels)
                if not 2 <= len(names) <= MAX_SCORE_LEVELS or len(set(names)) != len(names):
                    raise ValueError("a score needs 2-10 distinct ordered levels")
                packed.append(Question(question.instructions, list(names), 0, "score"))
            else:
                raise TypeError(f"unsupported question type for {identifier!r}")
            specs.append((identifier, question, names))
        example = DecisionExample("system-one", state, packed, "inference", "local", "inference")
        output = self.model(move_batch(self.collator([example]), self.device))
        answers: dict[str, Answer] = {}
        for (identifier, question, names), logits in zip(specs, output.logits, strict=True):
            probabilities = tuple((logits.float() / self.temperature).softmax(-1).cpu().tolist())
            best = max(range(len(names)), key=probabilities.__getitem__)
            if isinstance(question, NoulQuestion):
                answers[identifier] = Noul(
                    probabilities[0],
                    probabilities[1],
                    probabilities[2] if question.allow_unknown else None,
                    (True, False, None)[best],
                    probabilities[best],
                )
            elif isinstance(question, ScoreQuestion):
                values = tuple(float(index + 1) for index in range(len(names)))
                answers[identifier] = Score(
                    values,
                    probabilities,
                    sum(v * p for v, p in zip(values, probabilities, strict=True)),
                    values[best],
                    probabilities[best],
                )
            else:
                answers[identifier] = Choice(names, probabilities, names[best], probabilities[best])
        return answers

    def choice(
        self, state: str, instructions: str, criteria: Mapping[str, str] | Sequence[str]
    ) -> Choice:
        answer = self.evaluate(state, {"choice": ChoiceQuestion(instructions, criteria)})["choice"]
        assert isinstance(answer, Choice)
        return answer

    def gate(self, confidence: float) -> str:
        if not self.calibrated:
            return "verify"
        if confidence >= self.execute_threshold:
            return "execute"
        return "verify" if confidence >= self.verify_threshold else "escalate"

    def decision(self, state: str, question: str, options: Sequence[object]):
        """Drop-in for ``SemanticRouter.decision`` (options may be ``Candidate`` objects)."""
        from .semantic import RouteDecision

        criteria = {
            getattr(option, "name", str(option)): getattr(option, "description", str(option))
            for option in options
        }
        choice = self.choice(state, question, criteria)
        return RouteDecision(choice, self.gate(choice.confidence), self.calibrated)
