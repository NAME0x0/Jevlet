"""Reproducible training, evaluation, checkpointing, and hardware measurements."""

from __future__ import annotations

import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .calibration import fit_temperature
from .data import DecisionCollator, DecisionExample, JsonlDecisionDataset, permute_question
from .losses import decision_loss
from .metrics import compute_metrics, finite_metrics
from .model import JevletModel, ModelConfig


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    return torch.device(name)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def collect_predictions(
    model: JevletModel, loader: DataLoader, device: torch.device
) -> tuple[list[torch.Tensor], list[dict[str, Any]], dict[str, float]]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_records: list[dict[str, Any]] = []
    token_count = 0
    start = time.perf_counter()
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        all_logits.extend(logits.detach().cpu() for logits in output.logits)
        all_records.extend(output.records)
        token_count += int(batch["valid_mask"].sum())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return (
        all_logits,
        all_records,
        {
            "eval_seconds": elapsed,
            "questions_per_second": len(all_records) / elapsed,
            "tokens_per_second": token_count / elapsed,
            "latency_ms_per_question": elapsed * 1000 / max(len(all_records), 1),
        },
    )


@torch.no_grad()
def option_order_robustness(
    model: JevletModel,
    examples: list[DecisionExample],
    collator: DecisionCollator,
    device: torch.device,
    limit: int = 32,
) -> dict[str, float | int]:
    model.eval()
    differences = []
    agreements = []
    tested = 0
    for example in examples:
        for question in example.questions:
            if len(question.options) < 2 or tested >= limit:
                continue
            base_example = DecisionExample(
                example.example_id,
                example.state,
                [question],
                example.family,
                example.domain,
                example.split,
                example.is_ood,
            )
            permutation = list(reversed(range(len(question.options))))
            permuted_question = permute_question(question, permutation)
            permuted_example = DecisionExample(
                example.example_id + "-permuted",
                example.state,
                [permuted_question],
                example.family,
                example.domain,
                example.split,
                example.is_ood,
            )
            base = model(move_batch(collator([base_example]), device)).logits[0].softmax(-1).cpu()
            changed = (
                model(move_batch(collator([permuted_example]), device)).logits[0].softmax(-1).cpu()
            )
            restored = torch.empty_like(changed)
            for new_index, old_index in enumerate(permutation):
                restored[old_index] = changed[new_index]
            differences.append(float((base - restored).abs().max()))
            agreements.append(int(base.argmax() == restored.argmax()))
            tested += 1
    return {
        "option_order_examples": tested,
        "option_order_max_probability_delta": sum(differences) / max(tested, 1),
        "option_order_prediction_agreement": sum(agreements) / max(tested, 1),
    }


@torch.no_grad()
def benchmark_state_sharing(
    model: JevletModel,
    examples: list[DecisionExample],
    device: torch.device,
    max_seq_len: int,
    repeats: int = 5,
) -> dict[str, float | None]:
    candidates = [example for example in examples if len(example.questions) > 1]
    if not candidates:
        return {"shared_state_speedup": None, "incremental_ms_per_question": None}
    sample = candidates[0]
    timings = {}
    original_topology = model.config.attention_topology
    for topology in ("block_causal", "separate"):
        collator = DecisionCollator(max_seq_len=max_seq_len, attention_topology=topology)
        batch = move_batch(collator([sample]), device)
        model.config.attention_topology = topology
        model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[topology] = (time.perf_counter() - start) / repeats
    model.config.attention_topology = original_topology
    packed, separate = timings["block_causal"], timings["separate"]
    return {
        "shared_state_speedup": separate / max(packed, 1e-9),
        "incremental_ms_per_question": packed * 1000 / len(sample.questions),
    }


def save_checkpoint(
    path: str | Path,
    model: JevletModel,
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
    temperature: float,
    metrics: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "training_config": training_config,
            "temperature": temperature,
            "metrics": metrics,
        },
        destination,
    )


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[JevletModel, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = JevletModel(ModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


def train_experiment(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 1337))
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))
    model_config = ModelConfig.from_dict(config["model"])
    model = JevletModel(model_config).to(device)
    if device.type == "cuda" and config.get("gpu_memory_fraction"):
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(
            float(config["gpu_memory_fraction"]), device_index
        )
        torch.cuda.reset_peak_memory_stats(device)

    collator = DecisionCollator(
        max_seq_len=model_config.max_seq_len,
        attention_topology=model_config.attention_topology,
        max_state_bytes=int(config["data"].get("max_state_bytes", 128)),
        max_question_bytes=int(config["data"].get("max_question_bytes", 64)),
        max_option_bytes=int(config["data"].get("max_option_bytes", 48)),
    )
    train_dataset = JsonlDecisionDataset(config["data"]["train"])
    dev_dataset = JsonlDecisionDataset(config["data"]["dev"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 2)),
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=int(config["evaluation"].get("batch_size", 4)),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 3e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    use_amp = device.type == "cuda" and bool(config["training"].get("fp16", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gradient_accumulation = int(config["training"].get("gradient_accumulation", 1))
    max_steps = int(config["training"].get("max_steps", 100))
    max_seconds = config["training"].get("max_seconds")
    training_deadline = time.monotonic() + float(max_seconds) if max_seconds is not None else None
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    step = 0
    micro_step = 0
    questions_seen = 0
    tokens_seen = 0
    started = time.perf_counter()
    model.train()
    while step < max_steps:
        if training_deadline is not None and time.monotonic() >= training_deadline and step > 0:
            break
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        autocast = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
        with autocast:
            output = model(batch)
            loss, _ = decision_loss(
                output.logits,
                output.records,
                config["training"].get("loss", "ce"),
                float(config["training"].get("brier_weight", 0.25)),
                float(config["training"].get("label_smoothing", 0.05)),
            )
            scaled_loss = loss / gradient_accumulation
        scaler.scale(scaled_loss).backward()
        micro_step += 1
        questions_seen += len(output.records)
        tokens_seen += int(batch["valid_mask"].sum())
        if micro_step % gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = max(time.perf_counter() - started, 1e-9)

    logits, records, speed = collect_predictions(model, dev_loader, device)
    raw_metrics = compute_metrics(logits, records)
    temperature = fit_temperature(logits, records)
    calibrated = compute_metrics(logits, records, temperature)
    if not finite_metrics(raw_metrics) or not finite_metrics(calibrated):
        raise RuntimeError("non-finite evaluation metric")
    robustness = option_order_robustness(
        model,
        dev_dataset.examples,
        collator,
        device,
        int(config["evaluation"].get("permutation_examples", 32)),
    )
    sharing = benchmark_state_sharing(
        model,
        dev_dataset.examples,
        device,
        model_config.max_seq_len,
        int(config["evaluation"].get("benchmark_repeats", 5)),
    )
    metrics: dict[str, Any] = {
        **raw_metrics,
        **speed,
        **robustness,
        **sharing,
        "calibrated_ece": calibrated.get("ece"),
        "calibrated_nll": calibrated.get("nll"),
        "temperature": temperature,
        "parameter_count": model.parameter_count,
        "activated_parameter_count": model.parameter_count,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "train_seconds": train_seconds,
        "train_questions_per_second": questions_seen / train_seconds,
        "train_tokens_per_second": tokens_seen / train_seconds,
        "steps": step,
        "seed": seed,
        "device": str(device),
        "train_data_sha256": file_sha256(config["data"]["train"]),
        "dev_data_sha256": file_sha256(config["data"]["dev"]),
    }
    checkpoint_path = run_path / "best.pt"
    save_checkpoint(checkpoint_path, model, optimizer, config, temperature, metrics)
    (run_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_path / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics["checkpoint"] = str(checkpoint_path)
    return metrics
