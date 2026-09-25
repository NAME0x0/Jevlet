"""Export a checkpoint to ONNX for the C# app, verify it against PyTorch, and write goldens.

    python -m scripts.export_onnx data/daily/current.pt data/models/onnx/v4

Writes ``model.onnx``, ``vocab.txt`` and ``model.json`` to the output directory, and golden
files the C# tests replay (``app/tests/Jevlet.Core.Tests/golden``): tokenizer ids, packed rows
(ids, positions, branches, roles, option spans), and logits. Fails if ONNX Runtime and PyTorch
disagree by more than the tolerance on any case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jevlet.assistant import benchmark_v2 as bench
from jevlet.assistant.commands_synthetic import APP_POOL
from jevlet.assistant.skills import (
    SKILL_BY_KEY,
    SKILL_QUESTION,
    SKILLS,
    SLOTS,
    Environment,
    slot_options,
    window_label,
)
from jevlet.benchmarks import RISK_QUESTION, daily_state
from jevlet.data import DecisionExample, Question
from jevlet.onnx_export import export_onnx, graph_inputs, reference_logits
from jevlet.training import load_checkpoint

GOLDEN = Path("app/tests/Jevlet.Core.Tests/golden")
TOKENIZER_STRINGS = (
    "open spotify", "Hey Jevlet, can you open VS Code?", "what's 18% of 240", "café naïve résumé",
    "e-mail omar@northwind.ae re: Q3 (urgent!!)", "1,000.50 USD → AED", "日本語のテキスト", "emoji 😀👍 ok",
    "مرحبا بالعالم", "tab\tseparated\nnew line", "zero​width", "ALL CAPS SHOUTING", "C# and C++ and .NET",
    "it's 6:30pm, isn't it?", "Straße ÄÖÜ ß", "don't — can't — won't", "", "   ", "a" * 120,
    "supercalifragilisticexpialidocious", "pneumonoultramicroscopicsilicovolcanoconiosis x",
    "Ⅻ ① ﬁ ﬀ", "naïve café's crème brûlée", "hello world", "control\x07char", "Ωmega ωmega",
)  # fmt: skip


def _environment() -> Environment:
    return Environment(
        apps=sorted(APP_POOL, key=str.casefold),
        windows=[window_label(w.process, w.title) for w in bench.DESKTOP],
        current_window=window_label(bench.DESKTOP[0].process, bench.DESKTOP[0].title),
        events=list(bench.EVENTS), alarms=list(bench.ALARMS), todos=list(bench.TODOS),
        files=list(bench.FILES), reminders=list(bench.REMINDERS),
    )  # fmt: skip


def cases() -> list[DecisionExample]:
    env = _environment()
    names = [skill.name for skill in SKILLS]
    examples = []
    for index, case in enumerate(bench.CASES):
        state = daily_state(case.command, env.current_window)
        questions = [
            Question(SKILL_QUESTION, names, 0, "choice"),
            Question(RISK_QUESTION, ["True", "False"], 0, "noul"),
        ]
        skill = SKILL_BY_KEY[case.skill]
        for slot in skill.slots + skill.optional_slots:
            options = slot_options(slot, case.command, env)
            if len(options) > 1:
                questions.append(Question(SLOTS[slot].question, options, 0, "choice"))
        examples.append(
            DecisionExample(f"bench-{index}", state, questions, "golden", case.skill, "golden")
        )
    long_state = daily_state(" ".join(["remind me to call the bank"] * 120), "none")
    examples.append(
        DecisionExample(
            "long-state",
            long_state,
            [Question(SKILL_QUESTION, names, 0, "choice")],
            "golden",
            "x",
            "golden",
        )
    )
    for index, text in enumerate(TOKENIZER_STRINGS[:12]):
        examples.append(
            DecisionExample(
                f"stress-{index}", daily_state(text, "chrome: Docs"),
                [Question("Which option fits?", [text or "empty", "Not applicable", "café"], 0, "choice")],
                "golden", "x", "golden",
            )
        )  # fmt: skip
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--tolerance", type=float, default=2e-3)
    args = parser.parse_args()
    import onnxruntime as ort

    info = export_onnx(args.checkpoint, args.output)
    model, _ = load_checkpoint(args.checkpoint, "cpu")
    model = model.float().eval()
    session = ort.InferenceSession(
        str(Path(args.output) / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    rows, worst = [], 0.0
    for example in cases():
        inputs, records = graph_inputs(model, example)
        feeds = {name: tensor.numpy() for name, tensor in inputs.items()}
        onnx_logits = session.run(["logits"], feeds)[0]
        torch_logits = reference_logits(model, example)
        flat = np.concatenate([np.asarray(q, dtype=np.float32) for q in torch_logits])
        worst = max(worst, float(np.abs(flat - onnx_logits).max()))
        rows.append({
            "id": example.example_id,
            "state": example.state,
            "questions": [{"text": q.text, "options": q.options, "kind": q.kind} for q in example.questions],
            "input_ids": inputs["input_ids"][0].tolist(),
            "position_ids": inputs["position_ids"][0].tolist(),
            "records": [
                {"option_spans": r["option_spans"], "decide_position": r["decide_position"]} for r in records
            ],
            "logits": torch_logits,
        })  # fmt: skip
    if worst > args.tolerance:
        raise SystemExit(
            f"ONNX and PyTorch disagree by {worst:.2e} (tolerance {args.tolerance:.0e})"
        )
    tokenizer = model.tokenizer
    tokens = [
        {"text": text, "ids": tokenizer(text, add_special_tokens=False)["input_ids"]}
        for text in TOKENIZER_STRINGS
    ]
    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "tokenizer.json").write_text(
        json.dumps(tokens, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (GOLDEN / "packing.json").write_text(
        json.dumps({"model": info, "cases": rows}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(rows),
                "max_abs_logit_diff": worst,
                "onnx": str(Path(args.output) / "model.onnx"),
            }
        )
    )


if __name__ == "__main__":
    main()
