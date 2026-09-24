"""Ask a checkpoint which visible control of a live window fits a task. Never clicks.

python -m scripts.ground_live --checkpoint data/daily/current.pt --task "close this tab"
python -m scripts.ground_live --checkpoint ... --task "..." --window "Outlook"
"""

from __future__ import annotations

import argparse
import json
import time

from jevlet.desktop.native import NativeDesktop
from jevlet.grounding import actionable, ground
from jevlet.system_one import SystemOne


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--window", help="Substring of a window title (default: foreground)")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    desktop = NativeDesktop()
    engine = SystemOne(args.checkpoint)
    if args.window:
        matches = [
            w for w in desktop.list_windows() if args.window.casefold() in w.title.casefold()
        ]
        if not matches:
            raise SystemExit(f"no visible window title contains {args.window!r}")
        window = matches[0]
    else:
        window = desktop.active_window()
    if window is None:
        raise SystemExit("no foreground window")
    started = time.perf_counter()
    from jevlet.desktop.uia import FastUIA

    controls = FastUIA().read(window.handle)
    read_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    target, choice, risk = ground(engine, args.task, window.title, controls)
    decide_ms = (time.perf_counter() - started) * 1000
    ranked = sorted(
        zip(choice.options, choice.probabilities, strict=True), key=lambda pair: -pair[1]
    )
    print(
        json.dumps(
            {
                "window": window.title,
                "actionable_controls": len(actionable(controls)),
                "selected": choice.selected,
                "confidence": round(choice.confidence, 3),
                "risk": round(risk, 3),
                "rect": getattr(target, "rect", None),
                "top": [[option, round(p, 3)] for option, p in ranked[: args.top]],
                "uia_read_ms": round(read_ms, 1),
                "decision_ms": round(decide_ms, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
