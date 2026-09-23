"""Construct scratch (byte transformer) or pretrained (Jevlet-P) models behind one interface.

Both families return ``DecisionOutput`` from ``forward(batch)``, expose ``config.to_dict()``,
``config.attention_topology`` and ``parameter_count``, so training, metrics, calibration, and
the router do not care which backbone produced the logits.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .data import DecisionCollator
from .model import JevletModel, ModelConfig


def model_family(model_config: dict[str, Any]) -> str:
    family = model_config.get("family", "scratch")
    if family not in {"scratch", "pretrained"}:
        raise ValueError(f"unknown model family: {family}")
    return family


def build_model(model_config: dict[str, Any], *, load_weights: bool = True) -> nn.Module:
    if model_family(model_config) == "pretrained":
        from .pretrained import PretrainedConfig, PretrainedJevlet

        return PretrainedJevlet(PretrainedConfig.from_dict(model_config), load_weights=load_weights)
    return JevletModel(ModelConfig.from_dict(model_config))


def build_collator(
    model: nn.Module, data_config: dict[str, Any] | None = None, topology: str | None = None
) -> Any:
    """Rebuild preprocessing exactly from the model config (and scratch byte limits)."""
    if hasattr(model, "make_collator"):
        return model.make_collator(topology)
    data_config = data_config or {}
    return DecisionCollator(
        max_seq_len=model.config.max_seq_len,
        attention_topology=topology or model.config.attention_topology,
        max_state_bytes=int(data_config.get("max_state_bytes", 128)),
        max_question_bytes=int(data_config.get("max_question_bytes", 64)),
        max_option_bytes=int(data_config.get("max_option_bytes", 48)),
    )


def packed_topology(model: nn.Module) -> str:
    """The shared-state topology to benchmark against duplicated-state ``separate`` rows."""
    current = model.config.attention_topology
    if current != "separate":
        return current
    return "block_bidir" if hasattr(model, "make_collator") else "block_causal"
