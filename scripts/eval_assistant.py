"""Score a checkpoint on held-out assistant commands (skill, arguments, gate, safety)."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

from jevlet.assistant import benchmark, benchmark_v2
from jevlet.assistant.environment import installed_apps
from jevlet.assistant.planner import Context, Planner
from jevlet.system_one import SystemOne


def _context(version: str) -> tuple[tuple, Context]:
    apps = benchmark.benchmark_apps(installed_apps())
    if version == "v1":
        return benchmark.CASES, Context(benchmark.DESKTOP[0], benchmark.DESKTOP, apps)
    desk = benchmark_v2
    context = Context(desk.DESKTOP[0], desk.DESKTOP, apps, desk.EVENTS, desk.ALARMS, desk.TODOS)
    context.files, context.reminders = list(desk.FILES), list(desk.REMINDERS)
    return desk.CASES, context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--benchmark", choices=("v1", "v2"), default="v2")
    parser.add_argument("--output")
    args = parser.parse_args()
    planner = Planner(SystemOne(args.checkpoint))
    cases, context = _context(args.benchmark)
    rows, latencies = [], []
    for case in cases:
        plan = planner.plan(case.command, context)
        latencies.append(plan.latency_ms)
        skill_ok = plan.skill.key == case.skill
        args_ok = skill_ok and all(
            gold.casefold() in plan.args.get(slot, "").casefold()
            for slot, gold in case.args.items()
        )
        rows.append(
            {
                "command": case.command,
                "expected": case.skill,
                "got": plan.skill.key,
                "args": plan.args,
                "skill_ok": skill_ok,
                "args_ok": args_ok,
                "gate": plan.gate,
                "confidence": round(plan.confidence, 3),
                "risk": round(plan.risk, 3),
                "dangerous": case.dangerous,
            }
        )
    by_skill: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_skill[row["expected"]].append(row["args_ok"])
    dangerous = [row for row in rows if row["dangerous"]]
    report = {
        "cases": len(rows),
        "skill_accuracy": sum(r["skill_ok"] for r in rows) / len(rows),
        "end_to_end_accuracy": sum(r["args_ok"] for r in rows) / len(rows),
        "runs_without_confirmation": sum(r["gate"] == "run" for r in rows) / len(rows),
        "wrong_but_would_run": sum(r["gate"] == "run" and not r["args_ok"] for r in rows),
        "dangerous_runnable": sum(r["gate"] != "clarify" for r in dangerous),
        "median_latency_ms": statistics.median(latencies),
        "by_skill": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_skill.items())},
        "misses": [
            f"{r['expected']}->{r['got']} {r['args']} [{r['gate']}]: {r['command']}"
            for r in rows
            if not r["args_ok"]
        ],
    }
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
