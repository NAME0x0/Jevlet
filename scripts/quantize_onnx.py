"""Quantize an exported Jevlet ONNX model to int8 and check it against fp32 on the goldens.

    python -m scripts.quantize_onnx data/models/onnx/v4 data/models/onnx/v4-int8

Dynamic quantization (int8 weights, activations quantized at run time) of the MatMuls. The
check reports top-choice agreement and the largest probability change per question, using the
temperatures in model.json, over every golden case.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

GOLDEN = Path("app/tests/Jevlet.Core.Tests/golden/packing.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from jevlet.onnx_export import graph_inputs
    from jevlet.training import load_checkpoint

    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(source / "model.onnx"), str(output / "model.onnx"),
        weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
    )  # fmt: skip
    shutil.copyfile(source / "vocab.txt", output / "vocab.txt")
    info = json.loads((source / "model.json").read_text())
    info["quantization"] = "dynamic int8 (MatMul)"
    (output / "model.json").write_text(json.dumps(info, indent=2) + "\n")

    fp32 = ort.InferenceSession(str(source / "model.onnx"), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"])
    temperatures = info["temperatures"]
    model, _ = load_checkpoint(Path("data/daily") / info["source_checkpoint"], "cpu")
    from jevlet.data import DecisionExample, Question

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    agree = total = 0
    worst = 0.0
    times = {"fp32": [], "int8": []}
    for case in golden["cases"]:
        questions = [Question(q["text"], q["options"], 0, q["kind"]) for q in case["questions"]]
        example = DecisionExample(case["id"], case["state"], questions, "x", "x", "x")
        inputs, _ = graph_inputs(model, example)
        feeds = {name: tensor.numpy() for name, tensor in inputs.items()}
        outputs = {}
        for name, session in (("fp32", fp32), ("int8", int8)):
            started = time.perf_counter()
            outputs[name] = session.run(["logits"], feeds)[0]
            times[name].append((time.perf_counter() - started) * 1000)
        cursor = 0
        for question in questions:
            size = len(question.options)
            t = temperatures.get(question.kind, temperatures.get("default", 1.0))
            probs = {}
            for name in outputs:
                logits = outputs[name][cursor : cursor + size] / t
                exp = np.exp(logits - logits.max())
                probs[name] = exp / exp.sum()
            cursor += size
            agree += int(probs["fp32"].argmax() == probs["int8"].argmax())
            total += 1
            worst = max(worst, float(np.abs(probs["fp32"] - probs["int8"]).max()))
    report = {
        "questions": total,
        "top_choice_agreement": agree / total,
        "max_probability_change": worst,
        "median_ms": {name: round(float(np.median(values)), 1) for name, values in times.items()},
        "size_mb": {
            "fp32": round((source / "model.onnx").stat().st_size / 2**20, 1),
            "int8": round((output / "model.onnx").stat().st_size / 2**20, 1),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
