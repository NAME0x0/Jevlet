"""Tiny causal transformer with shared state and isolated decision branches."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import build_attention_mask
from .heads import make_decision_head
from .tokens import ByteTokenizer


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = ByteTokenizer.VOCAB_SIZE
    max_seq_len: int = 384
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: int = 8
    dropout: float = 0.1
    attention_topology: str = "block_causal"
    option_pool: str = "end"
    state_pool: str = "last"
    decision_head: str = "pointer"
    latent_pool_slots: int = 8
    gradient_checkpointing: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionOutput:
    logits: list[torch.Tensor]
    records: list[dict[str, Any]]
    hidden_states: torch.Tensor | None = None


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_width = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_width)
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~allowed[:, None, :, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous()
        return self.out(attended.view(batch, seq_len, width))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        hidden = config.d_model * config.ffn_mult
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), allowed)
        return x + self.mlp(self.mlp_norm(x))


class JevletModel(nn.Module):
    """A variable-choice model whose only output is a calibrated decision distribution."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.state_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.state_query = nn.Parameter(torch.randn(config.d_model) / math.sqrt(config.d_model))
        self.latent_queries = nn.Parameter(
            torch.randn(config.latent_pool_slots, config.d_model) / math.sqrt(config.d_model)
        )
        self.head = make_decision_head(config.decision_head, config.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _state_summary(
        self, hidden: torch.Tensor, branch_ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        state_mask = (branch_ids == -1) & valid_mask
        if self.config.state_pool == "last":
            indices = state_mask.long().sum(dim=1).sub(1).clamp_min(0)
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
        if self.config.state_pool == "mean":
            weights = state_mask.unsqueeze(-1)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        if self.config.state_pool == "attention":
            scores = torch.einsum("bld,d->bl", hidden, self.state_query)
            scores = scores.masked_fill(~state_mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores.float(), dim=-1).to(hidden.dtype)
            return torch.einsum("bl,bld->bd", weights, hidden)
        if self.config.state_pool == "latent":
            scores = torch.einsum("sd,bld->bsl", self.latent_queries, hidden)
            scores = scores / math.sqrt(hidden.shape[-1])
            scores = scores.masked_fill(~state_mask[:, None, :], torch.finfo(scores.dtype).min)
            weights = F.softmax(scores.float(), dim=-1).to(hidden.dtype)
            latents = torch.einsum("bsl,bld->bsd", weights, hidden)
            return latents.mean(dim=1)
        raise ValueError(f"unknown state pooling: {self.config.state_pool}")

    def _option_representation(
        self, hidden: torch.Tensor, row: int, record: dict[str, Any]
    ) -> torch.Tensor:
        representations = []
        for end_position, (start, stop) in zip(
            record["option_end_positions"], record["option_spans"], strict=True
        ):
            if self.config.option_pool == "end":
                value = hidden[row, end_position]
            elif self.config.option_pool == "last":
                value = hidden[row, max(start, stop - 1)]
            elif self.config.option_pool == "mean":
                value = hidden[row, start:stop].mean(dim=0)
            else:
                raise ValueError(f"unknown option pooling: {self.config.option_pool}")
            representations.append(value)
        return torch.stack(representations)

    def forward(self, batch: dict[str, Any], return_hidden: bool = False) -> DecisionOutput:
        token_ids = batch["token_ids"]
        position_ids = batch["position_ids"]
        branch_ids = batch["branch_ids"]
        valid_mask = batch["valid_mask"]
        if token_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("batch sequence exceeds configured max_seq_len")

        allowed = build_attention_mask(branch_ids, valid_mask, self.config.attention_topology)
        hidden = self.token_embedding(token_ids) + self.position_embedding(position_ids)
        hidden = self.embedding_dropout(hidden)
        hidden = hidden * valid_mask.unsqueeze(-1)

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, allowed, use_reentrant=False)
            else:
                hidden = block(hidden, allowed)
            hidden = hidden * valid_mask.unsqueeze(-1)
        hidden = self.final_norm(hidden)
        state_summary = self.state_projection(self._state_summary(hidden, branch_ids, valid_mask))

        logits: list[torch.Tensor] = []
        for record in batch["records"]:
            row = record["row"]
            decide = hidden[row, record["decide_position"]] + state_summary[row]
            options = self._option_representation(hidden, row, record)
            logits.append(self.head(decide, options))
        return DecisionOutput(logits, batch["records"], hidden if return_hidden else None)
