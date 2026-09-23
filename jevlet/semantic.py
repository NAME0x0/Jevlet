"""CPU-friendly pretrained, dynamic-option routing for day-one use.

The scratch transformer remains a separate research track. These probabilities are
not called calibrated until explicit feedback passes a held-out validation gate.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .api import Choice, Noul, Score
from .feedback import FeedbackStore, LabeledDecision

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "model_cache"


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    description: str


DEFAULT_ROUTING_CANDIDATES = (
    Candidate("Codex", "Write, edit, review, test, and debug code in a repository."),
    Candidate("Claude", "Analyze and write long-form text or explain complex documents."),
    Candidate("Gemini", "Analyze images, video, or multimodal documents."),
    Candidate("Retrieval", "Search files or the web for current, source-grounded information."),
    Candidate("Local", "Run a simple offline deterministic task on this computer."),
    Candidate("Human", "Ask a person to make a sensitive, uncertain, or consequential decision."),
)


@dataclass(frozen=True, slots=True)
class RoutingCase:
    state: str
    question: str
    expected: str | None  # None means the ambiguous request should be withheld.


ROUTING_BENCHMARK = (
    RoutingCase("A Python unit test fails and the source needs a fix.", "Who should act?", "Codex"),
    RoutingCase(
        "Find the latest official GPU driver documentation and cite it.",
        "Who should act?",
        "Retrieval",
    ),
    RoutingCase(
        "Rename a file on this laptop without using the network.", "Who should act?", "Local"
    ),
    RoutingCase(
        "Approve a large bank transfer to an unfamiliar person.", "Who should act?", "Human"
    ),
    RoutingCase("Summarize this long written report for me.", "Who should act?", "Claude"),
    RoutingCase("Describe the objects in this screenshot.", "Who should act?", "Gemini"),
    RoutingCase("Please do something with this.", "Who should act?", None),
)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    choice: Choice
    gate: str
    calibrated: bool


@dataclass(frozen=True, slots=True)
class AdaptationReport:
    promoted: bool
    reason: str
    train_count: int
    tune_count: int
    validation_count: int
    previous_nll: float | None = None
    candidate_nll: float | None = None


class SemanticRouter:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE,
        encoder: Any | None = None,
        feedback_store: FeedbackStore | None = None,
        adapter_path: str | Path | None = None,
        execute_threshold: float = 0.90,
        verify_threshold: float = 0.65,
    ) -> None:
        if not 0 < verify_threshold <= execute_threshold <= 1:
            raise ValueError("confidence thresholds must satisfy 0 < verify <= execute <= 1")
        if encoder is None:
            # Transformers' adapter probe does not always honor cache_folder.
            # Set the Hub cache before importing it, without overriding user config.
            os.environ.setdefault("HF_HOME", str(cache_dir))
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("install jevlet[semantic] for the pretrained router") from exc
            encoder = SentenceTransformer(model_name, device="cpu", cache_folder=str(cache_dir))
        self.encoder = encoder
        self._embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.model_name = model_name
        self.feedback_store = feedback_store
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.execute_threshold = execute_threshold
        self.verify_threshold = verify_threshold
        self._adapter = self._load_adapter()
        self._prototypes = self._load_prototypes()

    def _load_adapter(self) -> dict[str, Any]:
        if self.adapter_path is None or not self.adapter_path.exists():
            return {"alpha": 0.0, "temperature": 1.0, "train_ids": [], "validated_examples": 0}
        data = json.loads(self.adapter_path.read_text(encoding="utf-8"))
        if data.get("schema") != 1 or data.get("model") != self.model_name:
            raise ValueError("adapter schema or embedding model does not match")
        if data["alpha"] < 0 or data["temperature"] <= 0:
            raise ValueError("adapter has invalid scoring parameters")
        return data

    def _load_prototypes(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        ids = set(self._adapter["train_ids"])
        if not ids:
            return ()
        if self.feedback_store is None:
            raise ValueError("a feedback store is required to load a learned adapter")
        rows = [row for row in self.feedback_store.labeled_decisions() if row.id in ids]
        if len(rows) != len(ids):
            raise ValueError("adapter references missing feedback examples")
        return self._make_prototypes(rows)

    def _embed(self, texts: list[str]) -> torch.Tensor:
        missing = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
        if missing:
            vectors = self.encoder.encode(
                missing, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
            tensor = torch.as_tensor(vectors, dtype=torch.float32)
            if tensor.ndim != 2 or tensor.shape[0] != len(missing):
                raise ValueError("encoder must return one vector per input text")
            for text, vector in zip(missing, F.normalize(tensor, dim=1), strict=True):
                self._embedding_cache[text] = vector
        result = torch.stack([self._embedding_cache[text] for text in texts])
        for text in texts:
            self._embedding_cache.move_to_end(text)
        while len(self._embedding_cache) > 2048:
            self._embedding_cache.popitem(last=False)
        return result

    @staticmethod
    def _query(state: str, question: str) -> str:
        return f"Request: {question}\nContext: {state}"

    @staticmethod
    def _candidates(
        options: list[str | Candidate] | tuple[str | Candidate, ...],
    ) -> tuple[Candidate, ...]:
        candidates = tuple(
            option if isinstance(option, Candidate) else Candidate(option, option)
            for option in options
        )
        names = [option.name for option in candidates]
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("decisions need at least two uniquely named options")
        if any(not option.name.strip() or not option.description.strip() for option in candidates):
            raise ValueError("option names and descriptions cannot be empty")
        return candidates

    def _make_prototypes(
        self, rows: list[LabeledDecision]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if not rows:
            return ()
        queries = [self._query(row.state, row.question) for row in rows]
        descriptions = [dict(row.options)[row.label] for row in rows]
        vectors = self._embed(queries + descriptions)
        count = len(rows)
        return tuple((vectors[i], vectors[i + count]) for i in range(count))

    def _logits(
        self,
        state: str,
        question: str,
        candidates: tuple[Candidate, ...],
        *,
        alpha: float,
        prototypes: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        vectors = self._embed([self._query(state, question)] + [c.description for c in candidates])
        query, options = vectors[0], vectors[1:]
        logits = 10.0 * (options @ query)
        if alpha > 0 and prototypes:
            memory_queries = torch.stack([pair[0] for pair in prototypes])
            memory_options = torch.stack([pair[1] for pair in prototypes])
            query_match = ((memory_queries @ query - 0.60) / 0.40).clamp(0, 1)
            option_match = (options @ memory_options.T).clamp(0, 1)
            evidence = (option_match * query_match.unsqueeze(0)).max(dim=1).values
            logits = logits + alpha * evidence
        return logits

    @staticmethod
    def _choice(
        candidates: tuple[Candidate, ...], logits: torch.Tensor, temperature: float
    ) -> Choice:
        probabilities = (logits / temperature).softmax(dim=0).tolist()
        selected_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        names = tuple(candidate.name for candidate in candidates)
        return Choice(
            names, tuple(probabilities), names[selected_index], probabilities[selected_index]
        )

    def decision(
        self,
        state: str,
        question: str,
        options: list[str | Candidate] | tuple[str | Candidate, ...],
    ) -> RouteDecision:
        candidates = self._candidates(options)
        logits = self._logits(
            state,
            question,
            candidates,
            alpha=float(self._adapter["alpha"]),
            prototypes=self._prototypes,
        )
        choice = self._choice(candidates, logits, float(self._adapter["temperature"]))
        # A tiny acceptance slice can justify retaining an adapter, not
        # unattended action. Require a larger untouched slice for that gate.
        calibrated = self._adapter["validated_examples"] >= 20
        if not calibrated:
            gate = "verify"
        elif choice.confidence >= self.execute_threshold:
            gate = "execute"
        elif choice.confidence >= self.verify_threshold:
            gate = "verify"
        else:
            gate = "escalate"
        return RouteDecision(choice, gate, calibrated)

    def choice(
        self,
        state: str,
        question: str,
        options: list[str | Candidate] | tuple[str | Candidate, ...],
    ) -> Choice:
        return self.decision(state, question, options).choice

    def decision_two_stage(
        self,
        state: str,
        question: str,
        options: list[str | Candidate] | tuple[str | Candidate, ...],
        *,
        top_k: int = 16,
        listwise_reranker: Callable[[str, str, tuple[Candidate, ...]], Choice] | None = None,
    ) -> RouteDecision:
        """Independent top-k pruning, then optional external listwise reranking.

        Without a supplied trained reranker the second step remains semantic scoring,
        not a learned listwise model. Pruning invalidates full-set calibration, so
        the result always requires verification.
        """

        candidates = self._candidates(options)
        if top_k < 2:
            raise ValueError("top_k must be at least two")
        if len(candidates) <= top_k:
            return self.decision(state, question, candidates)
        logits = self._logits(
            state,
            question,
            candidates,
            alpha=float(self._adapter["alpha"]),
            prototypes=self._prototypes,
        )
        indices = torch.topk(logits, top_k).indices.tolist()
        shortlisted = tuple(candidates[index] for index in indices)
        if listwise_reranker is None:
            choice = self.choice(state, question, shortlisted)
        else:
            choice = listwise_reranker(state, question, shortlisted)
            if set(choice.options) != {candidate.name for candidate in shortlisted}:
                raise ValueError("reranker must return exactly the shortlisted options")
        return RouteDecision(choice, "verify", False)

    def noul(self, state: str, question: str, *, allow_unknown: bool = True) -> Noul:
        options = [
            Candidate("True", "The statement is supported by the provided context."),
            Candidate("False", "The statement is contradicted by the provided context."),
        ]
        if allow_unknown:
            options.append(Candidate("Unknown", "The context is insufficient to decide."))
        choice = self.choice(state, question, options)
        value = {"True": True, "False": False, "Unknown": None}[choice.selected]
        return Noul(
            choice.probabilities[0],
            choice.probabilities[1],
            choice.probabilities[2] if allow_unknown else None,
            value,
            choice.confidence,
        )

    def score(
        self,
        state: str,
        question: str,
        values: list[float] | tuple[float, ...],
        labels: list[str] | tuple[str, ...] | None = None,
    ) -> Score:
        if labels is None:
            labels = [str(value) for value in values]
        if len(values) != len(labels):
            raise ValueError("values and labels must have equal lengths")
        choice = self.choice(state, question, list(labels))
        expectation = sum(v * p for v, p in zip(values, choice.probabilities, strict=True))
        return Score(
            tuple(values),
            choice.probabilities,
            expectation,
            values[choice.options.index(choice.selected)],
            choice.confidence,
        )

    def _nll_accuracy(
        self,
        rows: list[LabeledDecision],
        alpha: float,
        temperature: float,
        prototypes: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[float, float]:
        losses = []
        correct = 0
        for row in rows:
            candidates = self._candidates([Candidate(*pair) for pair in row.options])
            logits = self._logits(
                row.state, row.question, candidates, alpha=alpha, prototypes=prototypes
            )
            label_index = [candidate.name for candidate in candidates].index(row.label)
            losses.append(-F.log_softmax(logits / temperature, dim=0)[label_index].item())
            correct += int(logits.argmax().item() == label_index)
        return sum(losses) / len(rows), correct / len(rows)

    def adapt(self, store: FeedbackStore, adapter_path: str | Path) -> AdaptationReport:
        """Promote memory weight/temperature only if a separate validation slice improves.

        IDs modulo five define stable train/tune/validation partitions. This is a
        conservative small-data guard, not a claim of domain-general calibration.
        """

        rows = store.labeled_decisions()
        train = [row for row in rows if row.id % 5 in (0, 1, 2)]
        tune = [row for row in rows if row.id % 5 == 3]
        validation = [row for row in rows if row.id % 5 == 4]
        counts = (len(train), len(tune), len(validation))
        if counts[0] < 8 or counts[1] < 3 or counts[2] < 3:
            return AdaptationReport(
                False, "need at least 8/3/3 labeled train/tune/validation decisions", *counts
            )
        candidate_prototypes = self._make_prototypes(train)
        best = (math.inf, 0.0, 1.0)
        for alpha in (0.0, 2.0, 4.0, 8.0):
            for temperature in (0.5, 1.0, 2.0, 4.0):
                nll, _ = self._nll_accuracy(tune, alpha, temperature, candidate_prototypes)
                if nll < best[0]:
                    best = (nll, alpha, temperature)
        _, alpha, temperature = best
        previous_nll, previous_accuracy = self._nll_accuracy(
            validation,
            float(self._adapter["alpha"]),
            float(self._adapter["temperature"]),
            self._prototypes,
        )
        candidate_nll, candidate_accuracy = self._nll_accuracy(
            validation, alpha, temperature, candidate_prototypes
        )
        if candidate_nll > previous_nll - 0.005 or candidate_accuracy < previous_accuracy:
            return AdaptationReport(
                False,
                "held-out NLL or accuracy did not improve",
                *counts,
                previous_nll,
                candidate_nll,
            )
        destination = Path(adapter_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "model": self.model_name,
            "alpha": alpha,
            "temperature": temperature,
            "train_ids": [row.id for row in train],
            "validated_examples": len(validation),
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
        self.feedback_store = store
        self.adapter_path = destination
        self._adapter = payload
        self._prototypes = candidate_prototypes
        return AdaptationReport(
            True,
            "promoted after held-out NLL and accuracy check",
            *counts,
            previous_nll,
            candidate_nll,
        )
