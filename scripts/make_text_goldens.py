"""Golden outputs of the text rules for the C# port (spans, courtesy, times, durations, shortlists).

    python -m scripts.make_text_goldens

Commands come from benchmark v2, generated commands (with typing noise), and edge cases.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from jevlet.assistant import benchmark_v2
from jevlet.assistant.commands_synthetic import APP_POOL, _command, _environment
from jevlet.assistant.skills import SKILLS, SLOTS, Environment, slot_options, window_label
from jevlet.assistant.text import (
    has_time,
    parse_duration,
    shortlist,
    similarity,
    span_candidates,
    strip_courtesy,
)

GOLDEN = Path("app/tests/Jevlet.Core.Tests/golden/text.json")


def commands() -> list[str]:
    # Only phrasing this repository owns: the goldens are committed under MIT, and the human
    # corpora (TOPv2 is CC BY-SA) must not be redistributed inside them.
    found = [case.command for case in benchmark_v2.CASES]
    rng = random.Random(2)
    for _ in range(4000):
        env, windows = _environment(rng, ())
        found.append(_command(rng, rng.choice(SKILLS).key, env, windows)[0])
    found += ["Straße nach Köln", "ŉ test", "type thank you", "remind me to call İsmail at 5", ""]
    return list(dict.fromkeys(found))


def main() -> None:
    apps = sorted(APP_POOL, key=str.casefold)
    rows = []
    for command in commands():
        rows.append({
            "command": command,
            "courtesy": strip_courtesy(command),
            "spans": span_candidates(command),
            "has_time": has_time(command),
            "duration": parse_duration(command),
            "shortlist": shortlist(command, apps, 8),
            "top_similarity": round(similarity(command, shortlist(command, apps, 1)[0]), 12),
        })  # fmt: skip
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(
        json.dumps({"apps": apps, "cases": rows}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {GOLDEN} ({len(rows)} commands)")
    write_slot_goldens(apps)


def write_slot_goldens(apps: list[str]) -> None:
    """Every slot's options for every benchmark command, in the benchmark's environment."""
    desktop = benchmark_v2.DESKTOP
    env = Environment(
        apps=apps,
        windows=[window_label(w.process, w.title) for w in desktop],
        current_window=window_label(desktop[0].process, desktop[0].title),
        events=list(benchmark_v2.EVENTS), alarms=list(benchmark_v2.ALARMS),
        todos=list(benchmark_v2.TODOS), files=list(benchmark_v2.FILES),
        reminders=list(benchmark_v2.REMINDERS),
    )  # fmt: skip
    rows = [
        {"command": case.command, "slot": slot, "options": slot_options(slot, case.command, env)}
        for case in benchmark_v2.CASES
        for slot in SLOTS
    ]
    target = GOLDEN.with_name("slots.json")
    environment = {
        "apps": env.apps, "windows": env.windows, "current_window": env.current_window,
        "events": env.events, "alarms": env.alarms, "todos": env.todos, "files": env.files,
        "reminders": env.reminders,
    }  # fmt: skip
    target.write_text(
        json.dumps({"environment": environment, "cases": rows}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {target} ({len(rows)} slot cases)")


if __name__ == "__main__":
    main()
