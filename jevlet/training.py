"""Reproducible training, evaluation, checkpointing, and hardware measurements."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .calibration import fit_temperature
from .data import DecisionExample, JsonlDecisionDataset, permute_question
from .families import build_collator, build_model, packed_topology
from .losses import decision_loss
from .metrics import compute_metrics, finite_metrics


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
    model: nn.Module, loader: DataLoader, device: torch.device
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
    model: nn.Module,
    examples: list[DecisionExample],
    collator: Any,
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
    model: nn.Module,
    examples: list[DecisionExample],
    device: torch.device,
    data_config: dict[str, Any] | None = None,
    repeats: int = 5,
) -> dict[str, float | None]:
    candidates = [example for example in examples if len(example.questions) > 1]
    if not candidates:
        return {"shared_state_speedup": None, "incremental_ms_per_question": None}
    sample = candidates[0]
    timings = {}
    original_topology = model.config.attention_topology
    packed = packed_topology(model)
    for topology in (packed, "separate"):
        collator = build_collator(model, data_config, topology)
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
    packed_seconds, separate_seconds = timings[packed], timings["separate"]
    return {
        "shared_state_speedup": separate_seconds / max(packed_seconds, 1e-9),
        "incremental_ms_per_question": packed_seconds * 1000 / len(sample.questions),
    }


def learning_rate_factor(step: int, training: dict[str, Any]) -> float:
    """Multiplier for the optimizer step after ``step`` completed steps.

    A pure function of the step count, so resumed runs follow the identical schedule.
    """
    schedule = training.get("lr_schedule", "constant")
    warmup = int(training.get("warmup_steps", 0))
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError("training.lr_schedule must be constant or cosine")
    horizon = max(1, int(training.get("max_steps", 100)) - warmup)
    progress = min(1.0, (step - warmup) / horizon)
    floor = float(training.get("min_lr_ratio", 0.1))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
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


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_progress(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    step: int,
    micro_step: int,
    epoch: int,
    batch_offset: int,
) -> None:
    """Write a recoverable snapshot at an optimizer boundary."""
    payload = {
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "training_config": config,
        "step": step,
        "micro_step": micro_step,
        "epoch": epoch,
        "batch_offset": batch_offset,
        "rng_state": _rng_state(),
        "train_data_sha256": file_sha256(config["data"]["train"]),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    # Pretrained backbones are rebuilt from config only; the checkpoint carries all weights.
    model = build_model(payload["model_config"], load_weights=False)
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
    init_from = config["training"].get("init_from")
    # Starting from a trained checkpoint needs its weights, not a fresh Hub download.
    model = build_model(config["model"], load_weights=not init_from)
    if init_from:
        initial = torch.load(init_from, map_location="cpu", weights_only=True)
        if initial["model_config"] != model.config.to_dict():
            raise ValueError("init_from checkpoint model configuration differs")
        model.load_state_dict(initial["model_state"])
    model = model.to(device)
    if device.type == "cuda" and config.get("gpu_memory_fraction"):
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(
            float(config["gpu_memory_fraction"]), device_index
        )
        torch.cuda.reset_peak_memory_stats(device)

    collator = build_collator(model, config["data"])
    train_dataset = JsonlDecisionDataset(config["data"]["train"], lazy=True)
    dev_dataset = JsonlDecisionDataset(config["data"]["dev"])
    eval_examples = dev_dataset.examples
    eval_limit = config["evaluation"].get("max_examples")
    if eval_limit is not None and int(eval_limit) < len(eval_examples):
        if int(eval_limit) < 1:
            raise ValueError("evaluation.max_examples must be positive")
        indices = sorted(random.Random(1729).sample(range(len(eval_examples)), int(eval_limit)))
        eval_examples = [eval_examples[index] for index in indices]
    if not len(train_dataset):
        raise ValueError("training dataset is empty")

    def train_loader_for_epoch(epoch: int) -> DataLoader:
        generator = torch.Generator().manual_seed(seed + epoch)
        return DataLoader(
            train_dataset,
            batch_size=int(config["training"].get("batch_size", 2)),
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collator,
        )

    dev_loader = DataLoader(
        eval_examples,
        batch_size=int(config["evaluation"].get("batch_size", 4)),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    learning_rate = float(config["training"].get("learning_rate", 3e-4))
    if hasattr(model, "parameter_groups"):
        backbone_rate = float(config["training"].get("backbone_learning_rate", learning_rate))
        parameters: Any = model.parameter_groups(learning_rate, backbone_rate)
    else:
        parameters = model.parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
    amp_dtype = config["training"].get("amp_dtype")
    if amp_dtype is None:
        amp_dtype = "fp16" if config["training"].get("fp16", True) else "none"
    if amp_dtype not in {"none", "fp16", "bf16"}:
        raise ValueError("training.amp_dtype must be none, fp16, or bf16")
    if device.type == "cuda" and amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is not supported by this CUDA device")
    use_amp = device.type == "cuda" and amp_dtype != "none"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == "fp16")
    gradient_accumulation = int(config["training"].get("gradient_accumulation", 1))
    max_steps = int(config["training"].get("max_steps", 100))
    max_seconds = config["training"].get("max_seconds")
    training_deadline = time.monotonic() + float(max_seconds) if max_seconds is not None else None
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    save_every_steps = int(config["training"].get("save_every_steps", 0))
    if save_every_steps < 0:
        raise ValueError("save_every_steps must be nonnegative")
    # Weights kept at these steps (never overwritten) for a scaling curve on held-out benchmarks.
    keep_steps = {int(value) for value in config["training"].get("keep_steps", ())}
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    epoch = 0
    batch_offset = 0
    resume_from = config["training"].get("resume_from")
    if resume_from:
        payload = torch.load(resume_from, map_location="cpu", weights_only=True)
        if payload["model_config"] != model.config.to_dict():
            raise ValueError("resume checkpoint model configuration differs")
        if payload["train_data_sha256"] != file_sha256(config["data"]["train"]):
            raise ValueError("resume checkpoint training data differs")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scaler.load_state_dict(payload["scaler_state"])
        step = int(payload["step"])
        micro_step = int(payload["micro_step"])
        epoch = int(payload["epoch"])
        batch_offset = int(payload["batch_offset"])
        if micro_step % gradient_accumulation:
            raise ValueError("resume checkpoint is not at an optimizer boundary")
        _restore_rng_state(payload["rng_state"])

    train_loader = train_loader_for_epoch(epoch)
    if batch_offset > len(train_loader):
        raise ValueError("resume checkpoint batch offset exceeds epoch")
    iterator = iter(train_loader)
    for _ in range(batch_offset):
        next(iterator)
    questions_seen = 0
    tokens_seen = 0
    started = time.perf_counter()
    first_step = step
    log_every = int(config["training"].get("log_every_steps", 100))
    model.train()
    while step < max_steps:
        if (
            training_deadline is not None
            and time.monotonic() >= training_deadline
            and step > 0
            and micro_step % gradient_accumulation == 0
        ):
            break
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            batch_offset = 0
            train_loader = train_loader_for_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        batch_offset += 1
        batch = move_batch(batch, device)
        torch_amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        autocast = torch.amp.autocast("cuda", dtype=torch_amp_dtype) if use_amp else nullcontext()
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
            factor = learning_rate_factor(step, config["training"])
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if log_every and step % log_every == 0:
                elapsed = time.perf_counter() - started
                rate = (step - first_step) / elapsed
                progress = {
                    "step": step,
                    "loss": round(float(loss), 4),
                    "lr_factor": round(factor, 4),
                    "steps_per_second": round(rate, 3),
                    "tokens_per_second": int(tokens_seen / elapsed),
                    "eta_hours": round((max_steps - step) / rate / 3600, 2),
                }
                print(json.dumps(progress), flush=True)
            if step in keep_steps:
                save_checkpoint(
                    run_path / f"step-{step}.pt", model, optimizer, config, 1.0, {"steps": step}
                )
            if save_every_steps and step % save_every_steps == 0:
                _save_progress(
                    run_path / "last.pt",
                    model,
                    optimizer,
                    scaler,
                    config,
                    step,
                    micro_step,
                    epoch,
                    batch_offset,
                )
    _save_progress(
        run_path / "last.pt",
        model,
        optimizer,
        scaler,
        config,
        step,
        micro_step,
        epoch,
        batch_offset,
    )
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
        eval_examples,
        collator,
        device,
        int(config["evaluation"].get("permutation_examples", 32)),
    )
    sharing = benchmark_state_sharing(
        model,
        eval_examples,
        device,
        config["data"],
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
        "amp_dtype": amp_dtype if use_amp else "none",
        "train_data_sha256": file_sha256(config["data"]["train"]),
        "dev_data_sha256": file_sha256(config["data"]["dev"]),
        "evaluation_examples": len(eval_examples),
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
