"""Glue between desktop suggestions and explicit router feedback."""

from __future__ import annotations

from jevlet.feedback import FeedbackStore
from jevlet.semantic import Candidate, RouteDecision, SemanticRouter

from .native import DesktopContext

ROUTE_DESCRIPTIONS = {
    "Local": "Perform a simple, deterministic, offline action in a local application.",
    "Codex": "Ask Codex to edit, test, debug, or review code in a repository.",
    "Claude": "Analyze and write long-form text or explain complex documents.",
    "Gemini": "Analyze images, video, or multimodal documents.",
    "Retrieval": "Retrieve up-to-date facts from local files or the web with sources.",
    "Human": "Ask a person to decide a sensitive, risky, or ambiguous question.",
}


class DesktopRoutingSession:
    def __init__(self, router: SemanticRouter, store: FeedbackStore) -> None:
        self.router = router
        self.store = store
        self.last_id: int | None = None

    def suggest(
        self, task: str, context: DesktopContext, options: tuple[str, ...]
    ) -> RouteDecision:
        window = context.window
        state = f"Window: {window.title}; process: {window.process_id}" if window else ""
        candidates = tuple(Candidate(name, ROUTE_DESCRIPTIONS.get(name, name)) for name in options)
        result = self.router.decision(state, task, candidates)
        self.last_id = self.store.log_decision(
            state,
            task,
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
