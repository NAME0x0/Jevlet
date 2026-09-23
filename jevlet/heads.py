"""Dynamic decision heads that score a variable number of text options."""

from __future__ import annotations

import math

import torch
from torch import nn


class PointerHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.scale = math.sqrt(width)

    def forward(self, decide: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        query = self.query(decide)
        keys = self.key(options)
        return torch.mv(keys, query) / self.scale


class BilinearPointerHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.weight = nn.Parameter(torch.empty(width, width))
        nn.init.xavier_uniform_(self.weight)
        self.scale = math.sqrt(width)

    def forward(self, decide: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        query = self.query(decide) @ self.weight
        return torch.mv(self.key(options), query) / self.scale


class CompatibilityHead(nn.Module):
    def __init__(self, width: int, kind: str) -> None:
        super().__init__()
        hidden = max(32, width // 2)
        input_width = width * 3
        if kind == "linear":
            self.network = nn.Linear(input_width, 1)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_width, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )
        else:
            raise ValueError(f"unknown compatibility head: {kind}")

    def forward(self, decide: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        expanded = decide.expand_as(options)
        features = torch.cat([expanded, options, (expanded - options).abs()], dim=-1)
        return self.network(features).squeeze(-1)


def make_decision_head(kind: str, width: int) -> nn.Module:
    if kind == "pointer":
        return PointerHead(width)
    if kind == "bilinear":
        return BilinearPointerHead(width)
    if kind in {"linear", "mlp"}:
        return CompatibilityHead(width, kind)
    raise ValueError(f"unknown decision head: {kind}")
