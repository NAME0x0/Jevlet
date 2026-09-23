"""Fixed day-one routing smoke set; not a substitute for user-domain validation."""

from __future__ import annotations

import argparse
import json
import time

from jevlet.semantic import DEFAULT_ROUTING_CANDIDATES, ROUTING_BENCHMARK, SemanticRouter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    kwargs = {"model_name": args.model} if args.model else {}
    router = SemanticRouter(**kwargs)
    records = []
    for case in ROUTING_BENCHMARK:
        started = time.perf_counter()
        decision = router.decision(case.state, case.question, DEFAULT_ROUTING_CANDIDATES)
        elapsed_ms = (time.perf_counter() - started) * 1000
        success = (
            decision.gate != "execute"
            if case.expected is None
            else decision.choice.selected == case.expected
        )
        records.append(
            {
                "expected": case.expected,
                "selected": decision.choice.selected,
                "gate": decision.gate,
                "correct": success,
                "latency_ms": round(elapsed_ms, 1),
            }
        )
    print(
        json.dumps(
            {
                "accuracy": sum(record["correct"] for record in records) / len(records),
                "mean_latency_ms": round(
                    sum(record["latency_ms"] for record in records) / len(records), 1
                ),
                "cases": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
