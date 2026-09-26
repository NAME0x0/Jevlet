"""Export a Jevlet-P checkpoint to ONNX for the C# app.

The graph takes what the collator produces for one packed row and returns one logit per option:

    input_ids      int64 [1, T]      packed state + question branches
    position_ids   int64 [1, T]      branch positions restart after the state
    attention_mask bool  [1, 1, T, T] build_pretrained_mask(...) for the row
    option_pool    float [O, T]      row o averages option o's body tokens ("mean" pooling)
    decide_select  float [O, T]      row o selects the [DECIDE] token of option o's question
    -> logits      float [O]         pointer head: key(option) . query(decide) / sqrt(width)

Pooling and selection as matrices keep the head's arithmetic inside the graph, so the app only
packs tokens and applies per-kind temperatures and a softmax per question. ``model.json``
carries everything the app needs to pack rows exactly like ``PretrainedCollator``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import DecisionExample, Question
from .pretrained import PretrainedJevlet, build_pretrained_mask
from .training import load_checkpoint


class DecisionGraph(nn.Module):
    def __init__(self, model: PretrainedJevlet) -> None:
        super().__init__()
        if model.config.decision_head != "pointer":
            raise ValueError("only the pointer head is exported")
        self.backbone = model.backbone
        self.query = model.head.query
        self.key = model.head.key
        self.scale = float(model.head.scale)
        self.offset = int(model.position_offset)

    def forward(self, input_ids, position_ids, attention_mask, option_pool, decide_select):
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids + self.offset,
            token_type_ids=torch.zeros_like(input_ids),
        ).last_hidden_state[0]
        options = option_pool @ hidden
        decide = decide_select @ hidden
        return (self.key(options) * self.query(decide)).sum(-1) / self.scale


def graph_inputs(model: PretrainedJevlet, example: DecisionExample) -> tuple[dict, list[dict]]:
    """The exact tensors the ONNX graph takes for one example, plus the collator's records."""
    batch = model.make_collator()([example])
    mask = build_pretrained_mask(
        batch["branch_ids"], batch["role_ids"], batch["option_ids"], batch["valid_mask"],
        batch["attention_topology"],
    )  # fmt: skip
    length = batch["input_ids"].shape[1]
    records = batch["records"]
    count = sum(len(record["options"]) for record in records)
    pool = torch.zeros(count, length)
    decide = torch.zeros(count, length)
    row = 0
    for record in records:
        for option_index, (start, stop) in enumerate(record["option_spans"]):
            if model.config.option_pool == "end":
                pool[row, record["option_end_positions"][option_index]] = 1.0
            else:
                pool[row, start:stop] = 1.0 / (stop - start)
            decide[row, record["decide_position"]] = 1.0
            row += 1
    inputs = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
        "attention_mask": mask,
        "option_pool": pool,
        "decide_select": decide,
    }
    return inputs, records


def catalogue_identity(catalogue: dict[str, Any]) -> dict[str, Any]:
    """What the app checks before trusting a model with its catalogue (catalogue_fingerprint)."""
    from .assistant.skills import catalogue_fingerprint
    from .benchmarks import RISK_QUESTION

    fingerprint = catalogue_fingerprint(
        catalogue,
        risk_question=RISK_QUESTION,
        state_format="Task: {command}\nActive window: {window}",
    )
    return {
        "version": int(catalogue["version"]),
        "fingerprint": fingerprint,
        "skills": [skill["name"] for skill in catalogue["skills"]],
    }


def manifest(model: PretrainedJevlet, payload: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    tokenizer = model.tokenizer
    collator = model.make_collator()
    config = model.config
    return {
        "format": "jevlet-onnx-1",
        "source_checkpoint": checkpoint.name,
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "backbone": config.backbone,
        "vocab": "vocab.txt",
        "lowercase": bool(getattr(tokenizer, "do_lower_case", True)),
        "token_ids": {
            "cls": tokenizer.cls_token_id,
            "pad": collator.pad_id,
            "unk": collator.unk_id,
            "state": collator.state_id,
            "question": collator.question_id,
            "option": collator.option_id,
            "end_option": collator.end_option_id,
            "decide": collator.decide_id,
        },
        "max_position": model.max_position,
        "max_packed_len": config.max_packed_len,
        "max_state_tokens": config.max_state_tokens,
        "max_question_tokens": config.max_question_tokens,
        "max_option_tokens": config.max_option_tokens,
        "attention_topology": config.attention_topology,
        "option_pool": config.option_pool,
        "temperature": float(payload.get("temperature", 1.0)),
        "temperatures": {k: float(v) for k, v in (payload.get("temperatures") or {}).items()},
        "calibration": payload.get("calibration"),
    }


def export_onnx(
    checkpoint: str | Path,
    output_dir: str | Path,
    opset: int = 18,
    catalogue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write model.onnx, vocab.txt, and model.json. ``catalogue`` is the skills.json the model was
    trained with; its identity goes into model.json so the app can refuse a mismatched model."""
    checkpoint = Path(checkpoint)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, payload = load_checkpoint(checkpoint, "cpu")
    model = model.float().eval()
    graph = DecisionGraph(model).eval()
    example = DecisionExample(
        "export",
        "Task: open spotify\nActive window: none",
        [Question("Which action?", ["Open an app", "Search the web", "Not applicable"], 0)],
        "export", "export", "export",
    )  # fmt: skip
    inputs, _ = graph_inputs(model, example)
    names = list(inputs)
    target = output / "model.onnx"
    torch.onnx.export(
        graph,
        tuple(inputs.values()),
        str(target),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "tokens"},
            "position_ids": {1: "tokens"},
            "attention_mask": {2: "tokens", 3: "tokens"},
            "option_pool": {0: "options", 1: "tokens"},
            "decide_select": {0: "options", 1: "tokens"},
            "logits": {0: "options"},
        },
        opset_version=opset,
        dynamo=False,
    )
    vocab_file = getattr(model.tokenizer, "vocab_file", None)
    vocab = Path(vocab_file) if vocab_file else None
    if vocab is None or not vocab.exists():
        model.tokenizer.save_vocabulary(str(output))
    else:
        shutil.copyfile(vocab, output / "vocab.txt")
    info = manifest(model, payload, checkpoint)
    if catalogue is not None:
        info["catalogue"] = catalogue_identity(catalogue)
    info["onnx_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def reference_logits(model: PretrainedJevlet, example: DecisionExample) -> list[list[float]]:
    """PyTorch logits per question, exactly as SystemOne computes them (before temperature)."""
    with torch.no_grad():
        output = model(model.make_collator()([example]))
    return [logits.float().tolist() for logits in output.logits]
