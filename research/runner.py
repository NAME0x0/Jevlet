"""Successive-halving autoresearch with immutable data checks and durable logs."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jevlet.training import train_experiment

from .pareto import pareto_frontier, promotion_order
from .search_space import CANDIDATE_SETS, Candidate


def _nested_set(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def apply_candidate(base: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    config = copy.deepcopy(base)
    for key, value in candidate.overrides.items():
        _nested_set(config, key, value)
    config["hypothesis"] = asdict(candidate)
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_research_data(config: dict[str, Any]) -> None:
    data_root = Path(config["data"]["train"]).parent
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("data/manifest.json is required; generate the fixed dataset first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "dev"):
        path = Path(config["data"][split])
        expected = manifest["splits"][split]["sha256"]
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"immutable {split} split hash changed: {actual} != {expected}")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _record_result(root: Path, log_path: Path, result: dict[str, Any]) -> None:
    result_path = root / result["run_id"] / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_jsonl(log_path, result)


def _cached_result(root: Path, run_id: str) -> dict[str, Any] | None:
    result_path = root / run_id / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("run_id") != run_id:
        raise RuntimeError(f"invalid cached result for {run_id}")
    if result.get("status") == "completed" and not (root / run_id / "last.pt").exists():
        raise RuntimeError(f"completed run {run_id} has no resumable checkpoint")
    return result


def run_successive_halving(config: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    verify_research_data(config)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = Path(config["research"].get("log", "research/experiment_log.jsonl"))
    candidate_set = config["research"].get("candidate_set", "night_one")
    if candidate_set not in CANDIDATE_SETS:
        raise ValueError(f"unknown research.candidate_set: {candidate_set}")
    candidates = CANDIDATE_SETS[candidate_set]()
    only = config["research"].get("candidates")
    if only:
        candidates = [candidate for candidate in candidates if candidate.candidate_id in only]
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    stage_steps = list(config["research"].get("stage_steps", [120, 600, 1800]))
    promotions = list(config["research"].get("promotions", [8, 3]))
    hours = float(config["research"].get("hours", 8.0))
    # Extending the budget of an interrupted run is not a configuration change.
    hashed = copy.deepcopy(config)
    extend_hours = hashed["research"].pop("extend_hours", None)
    config_hash = hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest()
    state_path = root / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["hours"] != hours or state["config_sha256"] != config_hash:
            raise ValueError("cannot resume research run with a different configuration")
        if extend_hours is not None:
            state["deadline"] = time.time() + float(extend_hours) * 3600
            state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    else:
        state = {"started_at": time.time(), "hours": hours, "config_sha256": config_hash}
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    deadline = state.get("deadline", state["started_at"] + hours * 3600)
    reserve_hours = float(config["research"].get("final_reserve_hours", 0.0))
    search_deadline = deadline - reserve_hours * 3600
    max_experiments = int(config["research"].get("max_experiments", 1_000_000))
    stage_max_seconds = list(config["research"].get("stage_max_seconds", []))
    stage_eval_examples = list(config["research"].get("stage_eval_examples", []))
    stage_permutation_examples = list(config["research"].get("stage_permutation_examples", []))
    all_results: list[dict[str, Any]] = []
    latest_stage_results: list[dict[str, Any]] = []
    active = candidates

    for stage_index, budget in enumerate(stage_steps, start=1):
        stage_results = []
        for candidate in active:
            run_id = f"stage-{stage_index}-{candidate.candidate_id}"
            cached = _cached_result(root, run_id)
            if cached is not None:
                all_results.append(cached)
                if cached["status"] == "completed":
                    stage_results.append(cached)
                continue
            if time.time() >= search_deadline or len(all_results) >= max_experiments:
                break
            run_config = apply_candidate(config, candidate)
            run_config["training"]["max_steps"] = int(budget)
            if stage_index <= len(stage_max_seconds):
                run_config["training"]["max_seconds"] = stage_max_seconds[stage_index - 1]
            if stage_index <= len(stage_eval_examples):
                run_config["evaluation"]["max_examples"] = stage_eval_examples[stage_index - 1]
            if stage_index <= len(stage_permutation_examples):
                run_config["evaluation"]["permutation_examples"] = stage_permutation_examples[
                    stage_index - 1
                ]
            run_config["seed"] = int(config.get("seed", 1337))
            if stage_index > 1:
                previous = root / f"stage-{stage_index - 1}-{candidate.candidate_id}" / "last.pt"
                if previous.exists():
                    run_config["training"]["resume_from"] = str(previous)
            interrupted = root / run_id / "last.pt"
            if interrupted.exists():
                run_config["training"]["resume_from"] = str(interrupted)
            message = (
                f"[{run_id}] Hypothesis: {candidate.hypothesis} "
                f"Expected mechanism: {candidate.expected_mechanism}"
            )
            print(message, flush=True)
            started = time.time()
            try:
                metrics = train_experiment(run_config, root / run_id)
                result = {
                    "run_id": run_id,
                    "stage": stage_index,
                    "budget_steps": budget,
                    "candidate": asdict(candidate),
                    "metrics": metrics,
                    "status": "completed",
                    "started_at": started,
                    "completed_at": time.time(),
                }
            except Exception as error:  # keep the night alive and record the exact failure
                result = {
                    "run_id": run_id,
                    "stage": stage_index,
                    "budget_steps": budget,
                    "candidate": asdict(candidate),
                    "status": "failed",
                    "error": repr(error),
                    "started_at": started,
                    "completed_at": time.time(),
                }
            _record_result(root, log_path, result)
            all_results.append(result)
            if result["status"] == "completed":
                stage_results.append(result)
        if not stage_results:
            break
        latest_stage_results = stage_results
        frontier = pareto_frontier(stage_results)
        (root / f"stage-{stage_index}-pareto.json").write_text(
            json.dumps(frontier, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if stage_index <= len(promotions):
            promoted = promotion_order(stage_results)[: int(promotions[stage_index - 1])]
            active = [by_id[result["candidate"]["candidate_id"]] for result in promoted]

    completed = [result for result in all_results if result["status"] == "completed"]
    final_results = []
    if completed and latest_stage_results and time.time() < deadline:
        search_winner = promotion_order(latest_stage_results)[0]
        winner_candidate = by_id[search_winner["candidate"]["candidate_id"]]
        for final_seed in config["research"].get("final_seeds", [config.get("seed", 1337)]):
            run_id = f"final-{winner_candidate.candidate_id}-seed-{final_seed}"
            cached = _cached_result(root, run_id)
            if cached is not None:
                all_results.append(cached)
                if cached["status"] == "completed":
                    final_results.append(cached)
                    completed.append(cached)
                continue
            if time.time() >= deadline:
                break
            final_config = apply_candidate(config, winner_candidate)
            final_config["model"].update(config["research"].get("final_model", {}))
            final_config["training"]["max_steps"] = int(
                config["research"].get("final_steps", stage_steps[-1])
            )
            remaining_seconds = max(1.0, deadline - time.time())
            final_config["training"]["max_seconds"] = remaining_seconds
            final_config["seed"] = int(final_seed)
            interrupted = root / run_id / "last.pt"
            if interrupted.exists():
                final_config["training"]["resume_from"] = str(interrupted)
            print(
                f"[{run_id}] Final verification. Hypothesis: {winner_candidate.hypothesis} "
                f"Expected mechanism: {winner_candidate.expected_mechanism}",
                flush=True,
            )
            started = time.time()
            try:
                metrics = train_experiment(final_config, root / run_id)
                result = {
                    "run_id": run_id,
                    "stage": "final",
                    "budget_steps": final_config["training"]["max_steps"],
                    "candidate": asdict(winner_candidate),
                    "metrics": metrics,
                    "status": "completed",
                    "started_at": started,
                    "completed_at": time.time(),
                }
            except Exception as error:
                result = {
                    "run_id": run_id,
                    "stage": "final",
                    "candidate": asdict(winner_candidate),
                    "status": "failed",
                    "error": repr(error),
                    "started_at": started,
                    "completed_at": time.time(),
                }
            _record_result(root, log_path, result)
            all_results.append(result)
            if result["status"] == "completed":
                final_results.append(result)
                completed.append(result)
    if not completed:
        summary = {"status": "failed", "experiments": len(all_results), "results": all_results}
    else:
        final_frontier = pareto_frontier(completed)
        winner_pool = final_results or latest_stage_results or completed
        winner = promotion_order(winner_pool)[0]
        summary = {
            "status": "completed",
            "experiments": len(all_results),
            "winner": winner,
            "pareto_frontier": final_frontier,
            "caveat": (
                "Night-one evidence is exploratory; repeat finalists across seeds before claims."
            ),
        }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
