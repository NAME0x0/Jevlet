"""Jevlet-P: a pretrained bidirectional encoder under the packed System-One topology.

The backbone replaces the scratch byte transformer; the decision interface does not change.
One row holds a shared state and every question branch. Branches see the state and their
own tokens, never a sibling, so packing is exactly equivalent to encoding each question
separately while the state is processed once. Branch position ids restart at the end of
the state, so only ``state + longest branch`` must fit the backbone's position table.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .attention import PAD_BRANCH, STATE_BRANCH
from .data import DecisionExample, Question
from .heads import make_decision_head
from .model import DecisionOutput

DEFAULT_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "model_cache"
STRUCTURE_TOKENS = ("[STATE]", "[QUESTION]", "[OPTION]", "[END_OPTION]", "[DECIDE]")
TOPOLOGIES = ("block_bidir", "option_isolated", "full", "separate")

ROLE_PAD = -1
ROLE_STATE = 0
ROLE_QUESTION = 1
ROLE_OPTION = 2
ROLE_DECIDE = 3


@dataclass(slots=True)
class PretrainedConfig:
    family: str = "pretrained"
    backbone: str = DEFAULT_BACKBONE
    attention_topology: str = "block_bidir"
    option_pool: str = "mean"
    decision_head: str = "pointer"
    max_packed_len: int = 1024
    max_state_tokens: int = 256
    max_question_tokens: int = 64
    max_option_tokens: int = 48
    gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PretrainedConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pretrained_mask(
    branch_ids: torch.Tensor,
    role_ids: torch.Tensor,
    option_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    topology: str,
) -> torch.Tensor:
    """Boolean ``[batch, 1, query, key]`` mask; True means the query may attend to the key.

    ``block_bidir``/``separate``: state sees state; a branch sees state plus its own branch.
    ``option_isolated``: additionally, an option sees only its own option tokens, question
    text sees no options, and only ``[DECIDE]`` compares options. With the collator's
    shared option positions this makes the output exactly invariant to option order.
    ``full``: every valid token sees every other valid token (sibling-leakage baseline).
    """

    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown pretrained attention topology: {topology}")
    valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
    if topology == "full":
        allowed = valid_pairs
    else:
        query_branch = branch_ids[:, :, None]
        key_branch = branch_ids[:, None, :]
        query_is_state = query_branch == STATE_BRANCH
        key_is_state = key_branch == STATE_BRANCH
        same_branch = (query_branch == key_branch) & (query_branch >= 0)
        allowed = valid_pairs & (
            (query_is_state & key_is_state) | (~query_is_state & (key_is_state | same_branch))
        )
        if topology == "option_isolated":
            query_role = role_ids[:, :, None]
            key_is_option = role_ids[:, None, :] == ROLE_OPTION
            same_option = option_ids[:, :, None] == option_ids[:, None, :]
            blocked = (
                same_branch
                & key_is_option
                & ((query_role == ROLE_QUESTION) | ((query_role == ROLE_OPTION) & ~same_option))
            )
            allowed = allowed & ~blocked
    seq_len = branch_ids.shape[1]
    eye = torch.eye(seq_len, dtype=torch.bool, device=branch_ids.device)
    allowed = allowed | (~valid_mask[:, :, None] & eye)
    return allowed[:, None, :, :]


def _load_tokenizer(backbone: str, cache_dir: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(backbone, cache_dir=str(cache_dir))
    tokenizer.add_special_tokens({"additional_special_tokens": list(STRUCTURE_TOKENS)})
    # Fields are truncated per budget after tokenization; the packed limit is enforced there.
    tokenizer.model_max_length = 1_000_000_000
    return tokenizer


class PretrainedCollator:
    """Pack one state and its question branches into token ids for a pretrained encoder."""

    def __init__(self, tokenizer: Any, config: PretrainedConfig, max_position: int) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.attention_topology = config.attention_topology
        self.max_position = max_position
        ids = tokenizer.convert_tokens_to_ids(list(STRUCTURE_TOKENS))
        self.state_id, self.question_id, self.option_id, self.end_option_id, self.decide_id = ids
        self.cls_id = tokenizer.cls_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    def _encode(self, texts: list[str]) -> list[list[int]]:
        """Token ids for [state, *questions, *options]. Questions and options repeat across
        rows (all 50 skill names in every command row), so they are tokenized once and cached;
        the state is always encoded fresh."""
        cache = self.__dict__.setdefault("_token_cache", {})
        unseen = list(dict.fromkeys(text for text in texts[1:] if text not in cache))
        if len(cache) + len(unseen) > 200_000:  # bound memory on pathological inputs
            cache.clear()
            unseen = list(dict.fromkeys(texts[1:]))
        if unseen:
            ids = self.tokenizer(unseen, add_special_tokens=False)["input_ids"]
            cache.update(zip(unseen, ids, strict=True))
        state = self.tokenizer(texts[:1], add_special_tokens=False)["input_ids"][0]
        return [state] + [cache[text] for text in texts[1:]]

    def _pack_row(
        self, example: DecisionExample, selected: list[int], topology: str
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[dict[str, Any]]]:
        questions: list[Question] = [example.questions[index] for index in selected]
        texts = [example.state] + [question.text for question in questions]
        for question in questions:
            texts.extend(question.options)
        encoded = self._encode(texts)
        prefix = [self.cls_id] if self.cls_id is not None else []
        # A long option list (50 skills) leaves less room for the state under the backbone's
        # position limit; truncate the state only as far as that branch needs.
        cursor = 1 + len(questions)
        extents = []
        for local_branch, question in enumerate(questions):
            segments = [
                min(len(encoded[cursor + i]), self.config.max_option_tokens) or 1
                for i in range(len(question.options))
            ]
            cursor += len(question.options)
            body = min(len(encoded[1 + local_branch]), self.config.max_question_tokens)
            if topology == "option_isolated":
                options = max(segments) + 2
            else:
                options = sum(segments) + 2 * len(segments)
            extents.append(1 + body + options + 1)
        room = self.max_position - len(prefix) - 1 - max(extents, default=0)
        state_body = encoded[0][: max(0, min(self.config.max_state_tokens, room))]
        token_ids = prefix + [self.state_id] + state_body
        state_len = len(token_ids)
        branch_ids = [STATE_BRANCH] * state_len
        role_ids = [ROLE_STATE] * state_len
        option_ids = [-1] * state_len
        position_ids = list(range(state_len))
        records: list[dict[str, Any]] = []
        cursor = 1 + len(questions)

        for local_branch, (question_index, question) in enumerate(
            zip(selected, questions, strict=True)
        ):
            question_body = encoded[1 + local_branch][: self.config.max_question_tokens]
            option_bodies = []
            for _ in question.options:
                body = encoded[cursor][: self.config.max_option_tokens] or [self.unk_id]
                option_bodies.append(body)
                cursor += 1
            offset = len(token_ids)
            branch_tokens = [self.question_id] + question_body
            branch_roles = [ROLE_QUESTION] * len(branch_tokens)
            branch_options = [-1] * len(branch_tokens)
            branch_positions = list(range(state_len, state_len + len(branch_tokens)))
            option_start_position = state_len + len(branch_tokens)
            spans: list[tuple[int, int]] = []
            ends: list[int] = []
            longest_option = 0
            for option_index, body in enumerate(option_bodies):
                segment = [self.option_id] + body + [self.end_option_id]
                start = offset + len(branch_tokens) + 1
                spans.append((start, start + len(body)))
                ends.append(offset + len(branch_tokens) + len(segment) - 1)
                if topology == "option_isolated":
                    first_position = option_start_position
                else:
                    first_position = state_len + len(branch_tokens)
                branch_positions.extend(range(first_position, first_position + len(segment)))
                branch_tokens.extend(segment)
                branch_roles.extend([ROLE_OPTION] * len(segment))
                branch_options.extend([option_index] * len(segment))
                longest_option = max(longest_option, len(segment))
            if topology == "option_isolated":
                decide_position = option_start_position + longest_option
            else:
                decide_position = state_len + len(branch_tokens)
            branch_tokens.append(self.decide_id)
            branch_roles.append(ROLE_DECIDE)
            branch_options.append(-1)
            branch_positions.append(decide_position)
            if max(branch_positions) >= self.max_position:
                raise ValueError(
                    f"example {example.example_id}: state plus question {question_index} needs "
                    f"position {max(branch_positions)}, over the backbone limit "
                    f"{self.max_position}; lower the token budgets"
                )
            token_ids.extend(branch_tokens)
            branch_ids.extend([local_branch] * len(branch_tokens))
            role_ids.extend(branch_roles)
            option_ids.extend(branch_options)
            position_ids.extend(branch_positions)
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
                    "option_spans": spans,
                    "option_end_positions": ends,
                    "decide_position": len(token_ids) - 1,
                }
            )
        if len(token_ids) > self.config.max_packed_len:
            raise ValueError(
                f"packed example {example.example_id} has {len(token_ids)} tokens, over "
                f"max_packed_len={self.config.max_packed_len}"
            )
        return token_ids, branch_ids, role_ids, option_ids, position_ids, records

    def __call__(self, examples: list[DecisionExample]) -> dict[str, Any]:
        topology = self.attention_topology
        rows = []
        for example in examples:
            if topology == "separate":
                rows.extend(
                    self._pack_row(example, [index], topology)
                    for index in range(len(example.questions))
                )
            else:
                rows.append(self._pack_row(example, list(range(len(example.questions))), topology))
        width = max(len(row[0]) for row in rows)
        columns: dict[str, list[list[int]]] = {
            "input_ids": [],
            "branch_ids": [],
            "role_ids": [],
            "option_ids": [],
            "position_ids": [],
        }
        valid_rows: list[list[bool]] = []
        records: list[dict[str, Any]] = []
        fills = (self.pad_id, PAD_BRANCH, ROLE_PAD, -1, 0)
        for row_index, row in enumerate(rows):
            padding = width - len(row[0])
            for key, values, fill in zip(columns, row[:5], fills, strict=True):
                columns[key].append(values + [fill] * padding)
            valid_rows.append([True] * len(row[0]) + [False] * padding)
            for record in row[5]:
                record["row"] = row_index
                records.append(record)
        batch: dict[str, Any] = {
            key: torch.tensor(values, dtype=torch.long) for key, values in columns.items()
        }
        batch["valid_mask"] = torch.tensor(valid_rows, dtype=torch.bool)
        batch["records"] = records
        batch["question_count"] = len(records)
        batch["example_count"] = len(examples)
        # Positions depend on topology, so the mask must come from the same topology.
        batch["attention_topology"] = topology
        return batch


class PretrainedJevlet(nn.Module):
    """Pretrained encoder + option-boundary pointer head; outputs only decision logits."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        load_weights: bool = True,
        cache_dir: str | Path = DEFAULT_CACHE,
    ) -> None:
        super().__init__()
        if config.attention_topology not in TOPOLOGIES:
            raise ValueError(f"unknown pretrained attention topology: {config.attention_topology}")
        if config.option_pool not in {"end", "mean"}:
            raise ValueError(f"unknown option pooling: {config.option_pool}")
        cache = Path(cache_dir)
        # Some Hub helpers ignore cache_dir; keep every download inside the project cache.
        os.environ.setdefault("HF_HOME", str(cache))
        from transformers import AutoConfig, AutoModel

        self.config = config
        self.tokenizer = _load_tokenizer(config.backbone, cache)
        options: dict[str, Any] = {"attn_implementation": "sdpa"}
        if load_weights:
            backbone = AutoModel.from_pretrained(
                config.backbone, cache_dir=str(cache), add_pooling_layer=False, **options
            )
        else:
            backbone_config = AutoConfig.from_pretrained(config.backbone, cache_dir=str(cache))
            backbone = AutoModel.from_config(backbone_config, add_pooling_layer=False, **options)
        original_vocab = backbone.get_input_embeddings().weight.shape[0]
        backbone.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        if load_weights:
            self._initialize_structure_tokens(backbone, original_vocab)
        if config.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.backbone = backbone
        # RoBERTa-style embeddings reserve positions 0..padding_idx; real tokens start after.
        offset_families = {"roberta", "xlm-roberta", "camembert", "mpnet"}
        self.position_offset = (
            int(backbone.config.pad_token_id) + 1
            if backbone.config.model_type in offset_families
            else 0
        )
        self.max_position = int(backbone.config.max_position_embeddings) - self.position_offset
        width = int(backbone.config.hidden_size)
        self.head = make_decision_head(config.decision_head, width)

    @torch.no_grad()
    def _initialize_structure_tokens(self, backbone: nn.Module, original_vocab: int) -> None:
        """Start markers from existing boundary embeddings rather than random vectors."""
        weight = backbone.get_input_embeddings().weight
        tokenizer = self.tokenizer
        anchors = {
            "[STATE]": tokenizer.sep_token_id,
            "[QUESTION]": tokenizer.sep_token_id,
            "[OPTION]": tokenizer.sep_token_id,
            "[END_OPTION]": tokenizer.sep_token_id,
            "[DECIDE]": tokenizer.cls_token_id,
        }
        mean = weight[:original_vocab].mean(dim=0)
        for token, anchor in anchors.items():
            index = tokenizer.convert_tokens_to_ids(token)
            if index < original_vocab:
                continue
            weight[index] = weight[anchor] if anchor is not None else mean

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def make_collator(self, topology: str | None = None) -> PretrainedCollator:
        config = self.config
        if topology is not None and topology != config.attention_topology:
            config = PretrainedConfig.from_dict(
                {**config.to_dict(), "attention_topology": topology}
            )
        return PretrainedCollator(self.tokenizer, config, self.max_position)

    def parameter_groups(self, learning_rate: float, backbone_learning_rate: float) -> list[dict]:
        backbone = [
            parameter for parameter in self.backbone.parameters() if parameter.requires_grad
        ]
        head = [parameter for parameter in self.head.parameters() if parameter.requires_grad]
        return [
            {"params": backbone, "lr": backbone_learning_rate},
            {"params": head, "lr": learning_rate},
        ]

    def forward(self, batch: dict[str, Any], return_hidden: bool = False) -> DecisionOutput:
        valid_mask = batch["valid_mask"]
        mask = build_pretrained_mask(
            batch["branch_ids"],
            batch["role_ids"],
            batch["option_ids"],
            valid_mask,
            batch.get("attention_topology", self.config.attention_topology),
        )
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": mask,
            "position_ids": batch["position_ids"] + self.position_offset,
        }
        if getattr(self.backbone.config, "type_vocab_size", 0):
            inputs["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        hidden = self.backbone(**inputs).last_hidden_state
        logits: list[torch.Tensor] = []
        for record in batch["records"]:
            row = record["row"]
            decide = hidden[row, record["decide_position"]]
            if self.config.option_pool == "end":
                options = hidden[row, record["option_end_positions"]]
            else:
                options = torch.stack(
                    [hidden[row, start:stop].mean(dim=0) for start, stop in record["option_spans"]]
                )
            logits.append(self.head(decide, options).float())
        return DecisionOutput(logits, batch["records"], hidden if return_hidden else None)
