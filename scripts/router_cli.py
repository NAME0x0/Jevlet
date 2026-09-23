"""Local pretrained routing with explicit feedback and conservative adaptation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from jevlet.feedback import FeedbackStore
from jevlet.semantic import DEFAULT_MODEL, DEFAULT_ROUTING_CANDIDATES, Candidate, SemanticRouter


def _candidate(raw: str) -> Candidate:
    name, separator, description = raw.partition("::")
    return Candidate(name.strip(), description.strip() if separator else name.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/feedback.sqlite3", help="Local SQLite feedback file")
    parser.add_argument("--adapter", default="data/router_adapter.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", help="Route with a Jevlet System-One checkpoint instead")
    commands = parser.add_subparsers(dest="command", required=True)
    route = commands.add_parser("route", help="Suggest a destination; never invokes it")
    route.add_argument("--state", default="")
    route.add_argument("--question", required=True)
    route.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="NAME::DESCRIPTION",
        help="Repeat for custom options; defaults to six routing profiles",
    )
    route.add_argument("--no-record", action="store_true", help="Do not store this request locally")
    feedback = commands.add_parser("feedback", help="Rate a prior decision")
    feedback.add_argument("id", type=int)
    rating = feedback.add_mutually_exclusive_group(required=True)
    rating.add_argument("--up", action="store_true")
    rating.add_argument("--down", action="store_true")
    feedback.add_argument("--correct", help="Correct option name; required for a downvote to train")
    commands.add_parser("adapt", help="Validate and promote a small feedback adapter if it helps")
    args = parser.parse_args()
    store = FeedbackStore(args.db)
    if args.command == "feedback":
        store.submit_feedback(args.id, approved=args.up, corrected_option=args.correct)
        print(json.dumps({"recorded": True, "training_label": args.up or args.correct is not None}))
        return
    if args.checkpoint:
        if args.command == "adapt":
            parser.error("adapt tunes the zero-shot router; retrain checkpoints instead")
        from jevlet.system_one import SystemOne

        router = SystemOne(args.checkpoint)
    else:
        router = SemanticRouter(
            model_name=args.model, feedback_store=store, adapter_path=args.adapter
        )
    if args.command == "adapt":
        report = router.adapt(store, args.adapter)
        print(json.dumps(asdict(report)))
        return
    options = (
        tuple(_candidate(raw) for raw in args.option) if args.option else DEFAULT_ROUTING_CANDIDATES
    )
    decision = router.decision(args.state, args.question, options)
    identifier = None
    if not args.no_record:
        identifier = store.log_decision(
            args.state,
            args.question,
            tuple((candidate.name, candidate.description) for candidate in options),
            decision.choice.selected,
        )
    print(
        json.dumps(
            {
                "id": identifier,
                "selected": decision.choice.selected,
                "gate": decision.gate,
                "calibrated": decision.calibrated,
                "probabilities": dict(
                    zip(decision.choice.options, decision.choice.probabilities, strict=True)
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
