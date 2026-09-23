"""Decision examples, JSONL loading, and topology-aware sequence packing."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .attention import PAD_BRANCH, STATE_BRANCH
from .tokens import ByteTokenizer


@dataclass(slots=True)
class Question:
    text: str
    options: list[str]
    label: int
    kind: str = "choice"
    target_probs: list[float] | None = None
    is_unknown: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Question:
        return cls(**values)


@dataclass(slots=True)
class DecisionExample:
    example_id: str
    state: str
    questions: list[Question]
    family: str
    domain: str
    split: str
    is_ood: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> DecisionExample:
        payload = dict(values)
        payload["questions"] = [Question.from_dict(item) for item in payload["questions"]]
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlDecisionDataset(Dataset[DecisionExample]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            self.examples = [DecisionExample.from_dict(json.loads(line)) for line in handle if line]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DecisionExample:
        return self.examples[index]


def write_jsonl(path: str | Path, examples: Iterable[DecisionExample]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")


class DecisionCollator:
    """Pack one state and its questions into a single branch-aware transformer sequence."""

    def __init__(
        self,
        tokenizer: ByteTokenizer | None = None,
        max_seq_len: int = 384,
        attention_topology: str = "block_causal",
        max_state_bytes: int = 128,
        max_question_bytes: int = 64,
        max_option_bytes: int = 48,
    ) -> None:
        self.tokenizer = tokenizer or ByteTokenizer()
        self.max_seq_len = max_seq_len
        self.attention_topology = attention_topology
        self.max_state_bytes = max_state_bytes
        self.max_question_bytes = max_question_bytes
        self.max_option_bytes = max_option_bytes

    def _question_tokens(
        self, question: Question
    ) -> tuple[list[int], list[tuple[int, int]], list[int]]:
        tokens = [self.tokenizer.QUESTION]
        tokens.extend(self.tokenizer.encode(question.text, self.max_question_bytes))
        spans: list[tuple[int, int]] = []
        end_positions: list[int] = []
        for option in question.options:
            tokens.append(self.tokenizer.OPTION)
            start = len(tokens)
            encoded = self.tokenizer.encode(option, self.max_option_bytes)
            if not encoded:
                encoded = self.tokenizer.encode(" ")
            tokens.extend(encoded)
            stop = len(tokens)
            tokens.append(self.tokenizer.END_OPTION)
            spans.append((start, stop))
            end_positions.append(len(tokens) - 1)
        tokens.append(self.tokenizer.DECIDE)
        return tokens, spans, end_positions

    def _pack_row(
        self, example: DecisionExample, selected_questions: list[int]
    ) -> tuple[list[int], list[int], list[int], list[dict[str, Any]]]:
        state_tokens = [self.tokenizer.STATE]
        state_tokens.extend(self.tokenizer.encode(example.state, self.max_state_bytes))
        token_ids = list(state_tokens)
        branch_ids = [STATE_BRANCH] * len(token_ids)
        position_ids = list(range(len(token_ids)))
        records: list[dict[str, Any]] = []

        for local_branch, question_index in enumerate(selected_questions):
            question = example.questions[question_index]
            branch, local_spans, local_ends = self._question_tokens(question)
            offset = len(token_ids)
            token_ids.extend(branch)
            branch_ids.extend([local_branch] * len(branch))
            position_ids.extend(range(len(state_tokens), len(state_tokens) + len(branch)))
            records.append(
                {
                    "row": -1,
                    "branch": local_branch,
                    "example_id": example.example_id,
                    "question_index": question_index,
                    "family": example.family,
                    "domain": example.domain,
                    "kind": question.kind,
                    "is_ood": example.is_ood,
                    "is_unknown": question.is_unknown,
                    "label": question.label,
                    "target_probs": question.target_probs,
                    "options": list(question.options),
                    "option_spans": [
                        (offset + start, offset + stop) for start, stop in local_spans
                    ],
                    "option_end_positions": [offset + position for position in local_ends],
                    "decide_position": len(token_ids) - 1,
                }
            )

        if len(token_ids) > self.max_seq_len:
            raise ValueError(
                f"packed example {example.example_id} has {len(token_ids)} tokens, "
                f"over max_seq_len={self.max_seq_len}; shorten fields or raise the limit"
            )
        return token_ids, branch_ids, position_ids, records

    def __call__(self, examples: list[DecisionExample]) -> dict[str, Any]:
        rows: list[tuple[list[int], list[int], list[int], list[dict[str, Any]]]] = []
        for example in examples:
            if self.attention_topology == "separate":
                rows.extend(
                    self._pack_row(example, [index]) for index in range(len(example.questions))
                )
            else:
                rows.append(self._pack_row(example, list(range(len(example.questions)))))

        max_length = max(len(row[0]) for row in rows)
        token_rows: list[list[int]] = []
        branch_rows: list[list[int]] = []
        position_rows: list[list[int]] = []
        valid_rows: list[list[bool]] = []
        records: list[dict[str, Any]] = []
        for row_index, (tokens, branches, positions, row_records) in enumerate(rows):
            padding = max_length - len(tokens)
            token_rows.append(tokens + [self.tokenizer.PAD] * padding)
            branch_rows.append(branches + [PAD_BRANCH] * padding)
            position_rows.append(positions + [0] * padding)
            valid_rows.append([True] * len(tokens) + [False] * padding)
            for record in row_records:
                record["row"] = row_index
                records.append(record)

        return {
            "token_ids": torch.tensor(token_rows, dtype=torch.long),
            "branch_ids": torch.tensor(branch_rows, dtype=torch.long),
            "position_ids": torch.tensor(position_rows, dtype=torch.long),
            "valid_mask": torch.tensor(valid_rows, dtype=torch.bool),
            "records": records,
            "question_count": len(records),
        }


def permute_question(question: Question, permutation: list[int]) -> Question:
    if sorted(permutation) != list(range(len(question.options))):
        raise ValueError("permutation must contain every option index exactly once")
    target_probs = None
    if question.target_probs is not None:
        target_probs = [question.target_probs[index] for index in permutation]
    return Question(
        text=question.text,
        options=[question.options[index] for index in permutation],
        label=permutation.index(question.label),
        kind=question.kind,
        target_probs=target_probs,
        is_unknown=question.is_unknown,
    )
