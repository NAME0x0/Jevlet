"""Deployable checkpoints: model weights and preprocessing config without optimizer state."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def export_checkpoint(source: str | Path, destination: str | Path, *, fp16: bool = False) -> Path:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload["model_state"]
    if fp16:
        state = {
            key: value.half() if value.is_floating_point() else value
            for key, value in state.items()
        }
    training = payload.get("training_config", {})
    slim = {
        "model_config": payload["model_config"],
        "model_state": state,
        "temperature": payload.get("temperature", 1.0),
        "temperatures": payload.get("temperatures", {}),
        "calibration": payload.get("calibration", {}),
        "metrics": payload.get("metrics", {}),
        "training_config": {"data": training.get("data", {}), "seed": training.get("seed")},
        "exported_from": str(Path(source).resolve()),
    }
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(slim, temporary)
    os.replace(temporary, target)
    return target
