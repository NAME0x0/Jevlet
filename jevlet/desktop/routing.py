"""Glue between desktop suggestions and explicit router feedback."""

from __future__ import annotations

from jevlet.benchmarks import RISK_QUESTION, ROUTE_QUESTION, daily_state
from jevlet.feedback import FeedbackStore
from jevlet.semantic import Candidate, RouteDecision

from .native import DesktopContext

ROUTE_DESCRIPTIONS = {
    "Local": "Perform a simple, deterministic, offline action in a local application.",
    "Codex": "Ask Codex to edit, test, debug, or review code in a repository.",
    "Claude": "Analyze and write long-form text or explain complex documents.",
    "Gemini": "Analyze images, video, or multimodal documents.",
    "Retrieval": "Retrieve up-to-date facts from local files or the web with sources.",
    "Human": "Ask a person to decide a sensitive, risky, or ambiguous question.",
}
# Jev's guidance sets a 0.9 bar for destructive operations: run unattended only when the model
# is at least 90% sure the task is safe, not merely when "risky" is below one half.
SAFE_TO_EXECUTE = 0.9


class DesktopRoutingSession:
    def __init__(self, router, store: FeedbackStore) -> None:
        self.router = router
        self.store = store
        self.last_id: int | None = None

    def suggest(
        self, task: str, context: DesktopContext, options: tuple[str, ...]
    ) -> RouteDecision:
        window = context.window.title if context.window else ""
        # The task lives in the state so every question branch can read it.
        state = daily_state(task, window)
        candidates = tuple(Candidate(name, ROUTE_DESCRIPTIONS.get(name, name)) for name in options)
        if hasattr(self.router, "evaluate"):
            from jevlet.system_one import ChoiceQuestion, NoulQuestion

            answers = self.router.evaluate(
                state,
                {
                    "route": ChoiceQuestion(
                        ROUTE_QUESTION, {c.name: c.description for c in candidates}
                    ),
                    "risk": NoulQuestion(RISK_QUESTION),
                },
            )
            choice, risk = answers["route"], answers["risk"].probability_true
            gate = self.router.gate(choice.confidence)
            if 1.0 - risk < SAFE_TO_EXECUTE and gate == "execute":
                gate = "verify"
            result = RouteDecision(choice, gate, self.router.calibrated, risk)
        else:
            result = self.router.decision(state, ROUTE_QUESTION, candidates)
        self.last_id = self.store.log_decision(
            state,
            ROUTE_QUESTION,
            tuple((candidate.name, candidate.description) for candidate in candidates),
            result.choice.selected,
        )
        return result

    def rate(self, rating: int, correction: str | None) -> None:
        if self.last_id is None:
            raise ValueError("Request a suggestion before rating it")
        identifier = self.last_id
        self.store.submit_feedback(identifier, approved=rating > 0, corrected_option=correction)
        self.last_id = None
