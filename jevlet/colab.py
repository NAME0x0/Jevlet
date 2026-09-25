"""Colab driver for Jevlet training: everything the notebook does, as testable functions.

Durable state lives on Google Drive: the data cache, the run mirror (``last.pt``,
``last.prev.pt``, step snapshots, progress), and the calibrated export. The VM holds only
working copies. Every function here is safe to call again after a crash, a browser disconnect,
a kernel restart, or a full runtime reset:

* data: restored from the Drive cache (hash-verified) or rebuilt, then cached;
* training: runs as a detached subprocess that survives kernel restarts; resumes from the
  newest readable checkpoint on Drive; out-of-memory halves the micro-batch (the effective
  batch is kept by gradient accumulation) and other crashes retry from the last save;
* publishing: calibration, evaluation, the model card, and the Hugging Face upload are
  idempotent and retried.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_FILES = (
    "mixture_v6/train.jsonl",
    "mixture_v6/dev.jsonl",
    "mixture_v6/vault/vault.jsonl",
    "mixture_v6/manifest.json",
    "real_commands/dev.jsonl",
    "real_commands/vault/vault.jsonl",
    "real_commands/vault/benchmark.jsonl",
    "real_commands/manifest.json",
    "commands_v6/dev.jsonl",
    "commands_v6/manifest.json",
)
OOM_MARKERS = ("OutOfMemoryError", "CUDA out of memory", "CUBLAS_STATUS_ALLOC_FAILED")


@dataclass(frozen=True)
class Paths:
    """Where things live. ``drive`` survives runtime resets; ``work`` does not."""

    drive: Path
    work: Path
    repo: Path
    version: str = "v6"

    @property
    def data(self) -> Path:
        return self.work / "jevlet_data"

    @property
    def cache(self) -> Path:
        return self.drive / "data_cache"

    @property
    def run(self) -> Path:
        return self.work / "runs" / self.version

    @property
    def mirror(self) -> Path:
        return self.drive / "runs" / self.version

    @property
    def export(self) -> Path:
        return self.drive / "exports" / self.version

    @property
    def pid_file(self) -> Path:
        return self.run / "train.pid"


def _sha256(path: Path) -> str:
    from .training import file_sha256

    return file_sha256(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- hardware


def gpu_profile() -> dict[str, Any]:
    """The GPU this session got and the micro-batch it can hold."""
    import torch

    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    if not torch.cuda.is_available():
        return {"name": "cpu", "memory_gb": 0.0, "bf16": False, "batch_size": 8, "workers": workers}
    props = torch.cuda.get_device_properties(0)
    memory = props.total_memory / 2**30
    batch = 64 if memory >= 22 else 32 if memory >= 14 else 16 if memory >= 7 else 8
    return {
        "name": props.name,
        "memory_gb": round(memory, 1),
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "batch_size": batch,
        "workers": workers,
    }


# --------------------------------------------------------------------------- data


def _verify(root: Path, index: dict[str, str]) -> bool:
    return all(
        (root / name).exists() and _sha256(root / name) == digest for name, digest in index.items()
    )


def save_data_cache(paths: Paths) -> Path:
    """Pack the files training and evaluation need into one archive on Drive."""
    paths.cache.mkdir(parents=True, exist_ok=True)
    index = {name: _sha256(paths.data / name) for name in CACHE_FILES}
    archive = paths.cache / f"data_{paths.version}.tar"
    partial = archive.with_name(archive.name + ".partial")
    with tarfile.open(partial, "w") as tar:  # JSONL compresses well but tar+copy is faster
        for name in CACHE_FILES:
            tar.add(paths.data / name, arcname=name)
    partial.replace(archive)
    _write_json(paths.cache / f"data_{paths.version}.index.json", index)
    return archive


def restore_data_cache(paths: Paths) -> bool:
    archive = paths.cache / f"data_{paths.version}.tar"
    index_path = paths.cache / f"data_{paths.version}.index.json"
    if not archive.exists() or not index_path.exists():
        return False
    index = json.loads(index_path.read_text())
    local = paths.work / archive.name
    shutil.copyfile(archive, local)  # read from Drive once, then extract locally
    with tarfile.open(local) as tar:
        if hasattr(tarfile, "data_filter"):  # Python >= 3.12 (and patched 3.11)
            tar.extractall(paths.data, filter="data")
        else:
            tar.extractall(paths.data)  # noqa: S202 - our own archive, hash-verified below
    local.unlink()
    if _verify(paths.data, index):
        return True
    _say("data cache failed verification; rebuilding")
    return False


def prepare_data(paths: Paths, workers: int) -> dict[str, Any]:
    """Make the v6 data available on the VM: verified local copy, Drive cache, or a rebuild."""
    index_path = paths.cache / f"data_{paths.version}.index.json"
    if index_path.exists() and _verify(paths.data, json.loads(index_path.read_text())):
        _say("data: local copy verified")
    elif restore_data_cache(paths):
        _say("data: restored from the Drive cache")
    else:
        _say("data: building from public sources (about 20-40 minutes on Colab CPUs)")
        subprocess.run(
            [
                sys.executable, "-m", "scripts.build_v6_data", "--root", str(paths.data),
                "--no-personal", "--workers", str(workers),
            ],
            cwd=paths.repo, check=True,
        )  # fmt: skip
        save_data_cache(paths)
        _say("data: built and cached on Drive")
    return json.loads((paths.data / "mixture_v6" / "manifest.json").read_text())


# --------------------------------------------------------------------------- training


def _readable(checkpoint: Path, train_sha: str) -> bool:
    import torch

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - any unreadable file means "try the older one"
        _say(f"{checkpoint.name} is unreadable ({type(error).__name__}); trying an older save")
        return False
    if payload.get("train_data_sha256") != train_sha:
        raise RuntimeError(
            f"{checkpoint} was trained on different data. Restore the Drive data cache, or set "
            "a new RUN_VERSION to start a separate run."
        )
    return True


def latest_resume(paths: Paths, train_sha: str) -> Path | None:
    """Newest readable checkpoint (Drive mirror first, then the VM), copied into the run dir."""
    paths.run.mkdir(parents=True, exist_ok=True)
    candidates = [paths.mirror / "last.pt", paths.mirror / "last.prev.pt", paths.run / "last.pt"]
    existing = sorted(
        (path for path in candidates if path.exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in existing:
        if _readable(candidate, train_sha):
            local = paths.run / "resume.pt"
            if candidate != local:
                shutil.copyfile(candidate, local)
            return local
    return None


def fit_batch(effective: int, limit: int) -> tuple[int, int]:
    """(micro-batch, accumulation) with micro-batch <= limit and product == effective."""
    micro = max(1, min(limit, effective))
    while effective % micro:
        micro -= 1
    return micro, effective // micro


def run_config(paths: Paths, profile: dict[str, Any], base: str) -> Path:
    """Resolve the training config for this session and store it on Drive.

    The recipe (steps, learning rates, effective batch, git commit) is fixed at the first
    session; only hardware-dependent fields (micro-batch, precision, workers) follow the GPU.
    """
    stored = paths.mirror / "run_config.json"
    if stored.exists():
        config = json.loads(stored.read_text())
    else:
        config = json.loads((paths.repo / base).read_text())
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=paths.repo, capture_output=True, text=True
        ).stdout.strip()
        config["provenance"] = {"git_commit": commit, "base_config": base, "created": time.time()}
    mixture = paths.data / "mixture_v6"
    config["data"].update(
        train=str(mixture / "train.jsonl"),
        dev=str(mixture / "dev.jsonl"),
        vault=str(mixture / "vault" / "vault.jsonl"),
    )
    training = config["training"]
    micro, accumulation = fit_batch(int(training["effective_batch"]), int(profile["batch_size"]))
    training.update(
        batch_size=micro,
        gradient_accumulation=accumulation,
        amp_dtype="bf16" if profile["bf16"] else "fp16",
        num_workers=int(profile["workers"]),
        mirror_dir=str(paths.mirror),
    )
    training.pop("resume_from", None)
    resume = latest_resume(paths, _sha256(mixture / "train.jsonl"))
    if resume is not None:
        training["resume_from"] = str(resume)
    config["evaluation"]["batch_size"] = micro
    config.setdefault("sessions", []).append(
        {"gpu": profile, "time": time.time(), "resume": bool(resume)}
    )
    _write_json(stored, config)
    local = paths.run / "run_config.json"
    _write_json(local, config)
    return local


def _alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a finished child is a zombie until reaped; treat it as gone
        waited, _ = os.waitpid(pid, os.WNOHANG)
        return waited == 0
    except ChildProcessError:
        return True  # not our child (kernel restarted): kill(0) said it exists


def running_pid(paths: Paths) -> int | None:
    if not paths.pid_file.exists():
        return None
    pid = int(paths.pid_file.read_text().strip() or 0)
    return pid if pid and _alive(pid) else None


def launch(paths: Paths, config_path: Path) -> int:
    """Start training detached from the kernel, so a kernel restart does not kill it."""
    existing = running_pid(paths)
    if existing:
        _say(f"training already running (pid {existing}); attaching")
        return existing
    log = (paths.run / "train.log").open("a", encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    command = [sys.executable, "-m", "scripts.train", "--config", str(config_path)]
    process = subprocess.Popen(
        [*command, "--run-dir", str(paths.run)],
        cwd=paths.repo, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )  # fmt: skip
    paths.pid_file.write_text(str(process.pid))
    _say(f"training started (pid {process.pid}); log: {paths.run / 'train.log'}")
    return process.pid


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _latest_progress(paths: Paths) -> dict[str, Any] | None:
    path = paths.run / "progress.jsonl"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1]) if lines else None


def monitor(paths: Paths, pid: int, poll_seconds: int = 30, mirror_seconds: int = 300) -> str:
    """Report progress until training ends. Returns "done", "failed", or "detached" (the user
    stopped the cell; training keeps running and this can be called again)."""
    last_step, last_mirror = None, 0.0
    try:
        while _alive(pid):
            progress = _latest_progress(paths)
            if progress and progress["step"] != last_step:
                last_step = progress["step"]
                _say(
                    f"step {progress['step']}  loss {progress['loss']:.4f}  "
                    f"{progress['tokens_per_second']:,} tok/s  ETA {progress['eta_hours']:.2f} h"
                )
            if time.time() - last_mirror > mirror_seconds:
                _copy_quietly(paths.run / "train.log", paths.mirror / "train.log")
                last_mirror = time.time()
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        _say("stopped watching; training continues in the background. Rerun the cell to watch.")
        return "detached"
    _copy_quietly(paths.run / "train.log", paths.mirror / "train.log")
    paths.pid_file.unlink(missing_ok=True)
    return "done" if (paths.run / "metrics.json").exists() else "failed"


def _copy_quietly(source: Path, target: Path) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    except OSError as error:
        _say(f"could not copy {source.name} to Drive: {error}")


def stop(paths: Paths) -> None:
    pid = running_pid(paths)
    if pid:
        if os.name == "nt":
            os.kill(pid, signal.SIGTERM)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        _say(f"stopped training (pid {pid}); the last save is on Drive")


def train(paths: Paths, profile: dict[str, Any], base: str, attempts: int = 5) -> str:
    """Train to completion, resuming and retrying through crashes."""
    if (paths.mirror / "metrics.json").exists() and (paths.mirror / "best.pt").exists():
        _say("training already finished for this run version")
        _copy_quietly(paths.mirror / "best.pt", paths.run / "best.pt")
        _copy_quietly(paths.mirror / "metrics.json", paths.run / "metrics.json")
        return "done"
    for attempt in range(1, attempts + 1):
        pid = running_pid(paths)
        if pid is None:
            pid = launch(paths, run_config(paths, profile, base))
        outcome = monitor(paths, pid)
        if outcome in {"done", "detached"}:
            return outcome
        tail = _tail(paths.run / "train.log")
        if any(marker in tail for marker in OOM_MARKERS) and profile["batch_size"] > 2:
            profile = {**profile, "batch_size": profile["batch_size"] // 2}
            _say(f"out of GPU memory: retrying with micro-batch {profile['batch_size']}")
        else:
            _say(f"training stopped with an error (attempt {attempt}/{attempts}):\n{tail}")
        time.sleep(10)
    raise RuntimeError("training kept failing; see train.log on Drive")


# --------------------------------------------------------------------------- evaluation


def evaluate_checkpoint(checkpoint: Path, data: Path, batch_size: int = 64) -> dict[str, Any]:
    """Accuracy and calibration per question type: skill, risk, and argument slots."""
    import torch
    from torch.utils.data import DataLoader

    from .data import JsonlDecisionDataset
    from .families import build_collator
    from .metrics import compute_metrics
    from .training import collect_predictions, load_checkpoint, resolve_device

    device = resolve_device("auto")
    model, payload = load_checkpoint(checkpoint, device)
    temperatures = payload.get("temperatures") or {}
    collator = build_collator(model, payload.get("training_config", {}).get("data"))
    examples = JsonlDecisionDataset(data).examples
    loader = DataLoader(examples, batch_size=batch_size, collate_fn=collator)
    logits, records, speed = collect_predictions(model, loader, device)
    groups: dict[str, list[int]] = {"all": list(range(len(records)))}
    for index, record in enumerate(records):
        family = record["family"]
        kind = (
            "skill"
            if record["question_index"] == 0
            else "risk"
            if record["question_index"] == 1
            else "slot"
        )
        groups.setdefault(f"{family}/{kind}" if family == "assistant" else family, []).append(index)
    report: dict[str, Any] = {"examples": len(examples), "speed": speed}
    for name, indices in sorted(groups.items()):
        subset_logits = [logits[i] for i in indices]
        subset_records = [records[i] for i in indices]
        kind = subset_records[0]["kind"] if subset_records else "choice"
        raw = compute_metrics(subset_logits, subset_records)
        report[name] = {
            "questions": raw["questions"],
            "accuracy": raw["accuracy"],
            "ece": raw["ece"],
        }
        # Per-kind temperatures only apply to a group of one kind (not to "all").
        if len({record["kind"] for record in subset_records}) == 1:
            temperature = float(temperatures.get(kind, payload.get("temperature", 1.0)))
            calibrated = compute_metrics(subset_logits, subset_records, temperature)
            report[name].update(calibrated_ece=calibrated["ece"], nll=calibrated["nll"])
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return report


def finish(paths: Paths) -> dict[str, Any]:
    """Calibrate, evaluate the final model and every snapshot, and write results to Drive."""
    paths.export.mkdir(parents=True, exist_ok=True)
    exported = paths.export / f"jevlet-{paths.version}.pt"
    best = paths.run / "best.pt"
    if not best.exists():
        _copy_quietly(paths.mirror / "best.pt", best)
    if not exported.exists():
        subprocess.run(
            [
                sys.executable, "-m", "scripts.calibrate", str(best), str(exported), "--fp16",
                "--data", str(paths.data / "real_commands" / "dev.jsonl"),
                "--data", str(paths.data / "commands_v6" / "dev.jsonl"),
            ],
            cwd=paths.repo, check=True,
        )  # fmt: skip
    results_path = paths.export / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    vault = paths.data / "real_commands" / "vault" / "vault.jsonl"
    dev = paths.data / "mixture_v6" / "dev.jsonl"
    if "final" not in results:
        results["final"] = {
            "real_vault": evaluate_checkpoint(exported, vault),
            "mixture_dev": evaluate_checkpoint(exported, dev),
        }
        _write_json(results_path, results)
    scaling = results.setdefault("scaling", {})
    for snapshot in sorted(paths.mirror.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1])):
        if snapshot.stem not in scaling:
            _copy_quietly(snapshot, paths.run / snapshot.name)
            scaling[snapshot.stem] = evaluate_checkpoint(paths.run / snapshot.name, vault)
            _write_json(results_path, results)
    import torch

    exported_payload = torch.load(exported, map_location="cpu", weights_only=True)
    results["temperatures"] = exported_payload.get("temperatures")
    results["calibration"] = exported_payload.get("calibration")
    run_config = json.loads((paths.mirror / "run_config.json").read_text())
    results["training"] = {
        "metrics": json.loads((paths.mirror / "metrics.json").read_text()),
        "config": run_config,
        "data_manifest": json.loads((paths.data / "mixture_v6" / "manifest.json").read_text()),
    }
    _write_json(results_path, results)
    for name in ("progress.jsonl", "run_config.json", "metrics.json"):
        _copy_quietly(paths.mirror / name, paths.export / name)
    _say(f"results written to {results_path}")
    return results


# --------------------------------------------------------------------------- publishing


def _retry(action, what: str, attempts: int = 5):
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:  # noqa: BLE001 - network errors come in many types
            if attempt == attempts:
                raise
            wait = 10 * 2 ** (attempt - 1)
            _say(f"{what} failed ({type(error).__name__}: {error}); retrying in {wait}s")
            time.sleep(wait)
    return None


def safetensors_export(checkpoint: Path, folder: Path) -> None:
    """Weights as safetensors plus a JSON config, for loading without pickle."""
    import torch
    from safetensors.torch import save_file

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {name: tensor.contiguous() for name, tensor in payload["model_state"].items()}
    save_file(state, str(folder / "model.safetensors"), metadata={"format": "pt"})
    config = {
        "model_config": payload["model_config"],
        "temperatures": payload.get("temperatures"),
        "calibration": payload.get("calibration"),
    }
    _write_json(folder / "config.json", config)


def publish(
    paths: Paths, repo_id: str, token: str, private: bool, github: str, license: str = "mit"
) -> str:
    """Upload weights, catalogues, results, and the model card; tag the version."""
    from huggingface_hub import HfApi

    from .model_card import render_model_card

    staging = paths.work / "hf_upload"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    exported = paths.export / f"jevlet-{paths.version}.pt"
    shutil.copyfile(exported, staging / exported.name)
    safetensors_export(exported, staging)
    for name in ("results.json", "progress.jsonl", "run_config.json"):
        if (paths.export / name).exists():
            shutil.copyfile(paths.export / name, staging / name)
    (staging / "shared").mkdir()
    for name in ("skills.json", "text_rules.json"):
        shutil.copyfile(paths.repo / "shared" / name, staging / "shared" / name)
    results = json.loads((paths.export / "results.json").read_text())
    (staging / "README.md").write_text(
        render_model_card(
            results, repo_id=repo_id, version=paths.version, github=github, license=license
        ),
        encoding="utf-8",
    )
    api = HfApi(token=token)
    _retry(lambda: api.create_repo(repo_id, private=private, exist_ok=True), "create repo")
    _retry(
        lambda: api.upload_folder(
            repo_id=repo_id,
            folder_path=str(staging),
            commit_message=f"Jevlet {paths.version}: weights, calibration, results, model card",
        ),
        "upload",
    )
    _retry(lambda: api.create_tag(repo_id, tag=paths.version, exist_ok=True), "tag")
    url = f"https://huggingface.co/{repo_id}"
    _write_json(paths.export / "published.json", {"repo": repo_id, "url": url, "time": time.time()})
    _say(f"published to {url}")
    return url
