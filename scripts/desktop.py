"""Run the local Windows Jevlet panel: python -m scripts.desktop."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from jevlet.desktop.app import DesktopApp
from jevlet.desktop.feedback import DesktopFeedbackStore
from jevlet.desktop.harness import DesktopHarness
from jevlet.desktop.routing import DesktopRoutingSession
from jevlet.feedback import FeedbackStore
from jevlet.semantic import DEFAULT_MODEL, SemanticRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Jevlet local desktop control panel")
    parser.add_argument("--feedback-db", type=Path, default=Path("data/desktop_feedback.sqlite3"))
    parser.add_argument("--router-db", type=Path, default=Path("data/feedback.sqlite3"))
    parser.add_argument("--adapter", type=Path, default=Path("data/router_adapter.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--checkpoint", type=Path, help="Jevlet System-One checkpoint (default: zero-shot router)"
    )
    parser.add_argument("--manual", action="store_true", help="Open desktop controls without model")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The desktop panel requires an interactive Windows session")
    decide = None
    on_rate = None
    if not args.manual:
        if sys.stdout is not None:
            print("Loading the local routing model; first use may download it.", flush=True)
        store = FeedbackStore(args.router_db)
        if args.checkpoint:
            from jevlet.system_one import SystemOne

            router = SystemOne(str(args.checkpoint))
        else:
            router = SemanticRouter(
                model_name=args.model, feedback_store=store, adapter_path=args.adapter
            )
        session = DesktopRoutingSession(router, store)
        decide = session.suggest
        on_rate = session.rate

    DesktopApp(DesktopHarness(), DesktopFeedbackStore(args.feedback_db), decide, on_rate).run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_path = Path("data/desktop_startup_error.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        raise
