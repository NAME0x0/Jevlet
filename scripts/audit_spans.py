"""Audit free-text span extraction: generator miss rate per skill and benchmark v2 recall.

A generated command whose gold text is not among ``span_candidates`` becomes a training row
with an inserted option the live app would never offer; a benchmark case whose gold is not a
substring of any candidate cannot be answered however good the model is.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from jevlet.assistant.benchmark_v2 import CASES
from jevlet.assistant.commands_synthetic import _command, _environment
from jevlet.assistant.skills import SKILLS
from jevlet.assistant.text import span_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    rng = random.Random(7)
    total, miss = Counter(), Counter()
    shown: dict[str, list[str]] = defaultdict(list)
    for _ in range(args.samples):
        skill = rng.choice(SKILLS).key
        env, windows = _environment(rng, ())
        command, gold, _ = _command(rng, skill, env, windows)
        if "text" not in gold:
            continue
        total[skill] += 1
        if gold["text"] not in span_candidates(command):
            miss[skill] += 1
            if len(shown[skill]) < args.examples:
                shown[skill].append(f"{command!r} -> {gold['text']!r}")
    overall = sum(miss.values()) / max(sum(total.values()), 1)
    print(f"generator text miss rate: {overall:.2%}")
    for skill in sorted(total):
        print(f"  {skill:16} {miss[skill]:5}/{total[skill]:<5} {miss[skill] / total[skill]:.1%}")
        for line in shown[skill]:
            print(f"      {line}")
    cases = [case for case in CASES if "text" in case.args]
    missed = [
        case
        for case in cases
        if not any(
            case.args["text"].casefold() in c.casefold() for c in span_candidates(case.command)
        )
    ]
    print(f"benchmark v2 text recall: {len(cases) - len(missed)}/{len(cases)}")
    for case in missed:
        candidates = span_candidates(case.command)
        print(f"  miss: {case.command!r} -> {case.args['text']!r}: {candidates}")


if __name__ == "__main__":
    main()
