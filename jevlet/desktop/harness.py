"""Preview and explicit confirmation boundary for desktop actions."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .native import DesktopContext, NativeDesktop


@dataclass(frozen=True, slots=True)
class ActionRequest:
    kind: str
    target: str = ""
    text: str = ""


@dataclass(frozen=True, slots=True)
class ActionProposal:
    proposal_id: str
    request: ActionRequest
    description: str
    expected_window_handle: int | None
    expected_window_title: str | None


@dataclass(frozen=True, slots=True)
class ActionResult:
    success: bool
    detail: str


class DesktopHarness:
    def __init__(self, backend: NativeDesktop | None = None) -> None:
        self.backend = backend or NativeDesktop()
        self._pending: ActionProposal | None = None

    def observe(self) -> DesktopContext:
        return self.backend.observe()

    def propose(
        self, request: ActionRequest, *, expected_window_handle: int | None = None
    ) -> ActionProposal:
        target_window = None
        if request.kind == "launch":
            if request.target not in self.backend.APPS:
                raise ValueError("Choose an allowlisted app")
            description = f"Open {request.target}"
        elif request.kind == "switch":
            if not request.target.strip():
                raise ValueError("Enter a window title")
            matches = [
                window
                for window in self.backend.list_windows()
                if request.target.casefold() in window.title.casefold()
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one matching window, found {len(matches)}")
            target_window = matches[0]
            description = f"Switch to {target_window.title!r}"
        elif request.kind == "focus":
            if not request.target.strip():
                raise ValueError("Enter an accessibility automation ID")
            description = f"Focus control {request.target!r}"
        elif request.kind == "type":
            if not request.text or len(request.text) > 4000:
                raise ValueError("Enter 1–4000 characters")
            description = f"Type {len(request.text)} characters into the foreground control"
        else:
            raise ValueError(f"Unsupported action: {request.kind}")
        if request.kind in {"focus", "type"}:
            handle = expected_window_handle
            if handle is None:
                current = self.backend.active_window()
                handle = current.handle if current else None
            target_window = next(
                (window for window in self.backend.list_windows() if window.handle == handle),
                None,
            )
            if target_window is None:
                raise RuntimeError("Target window is not available for preview")
        proposal = ActionProposal(
            uuid4().hex,
            request,
            description,
            target_window.handle if target_window else None,
            target_window.title if target_window else None,
        )
        self._pending = proposal
        return proposal

    def execute(self, proposal: ActionProposal, *, confirmed: bool = False) -> ActionResult:
        if not confirmed:
            raise PermissionError("Explicit confirmation is required")
        if proposal != self._pending:
            raise PermissionError("Proposal is stale or was not previewed")
        self._pending = None  # one-shot, including failed attempts
        request = proposal.request
        if request.kind in {"switch", "focus", "type"}:
            if proposal.expected_window_handle is None:
                raise RuntimeError("No target window was captured during preview")
            matches = [
                window
                for window in self.backend.list_windows()
                if window.handle == proposal.expected_window_handle
                and window.title == proposal.expected_window_title
            ]
            if len(matches) != 1:
                raise RuntimeError("Target window changed after preview; preview again")
            self.backend.activate_window(proposal.expected_window_handle)
            current = self.backend.active_window()
            if current is None or current.handle != proposal.expected_window_handle:
                raise RuntimeError("Foreground window changed after preview; preview again")
        if request.kind == "launch":
            self.backend.launch_app(request.target)
        elif request.kind == "focus":
            self.backend.focus_control(request.target)
        elif request.kind == "type":
            self.backend.type_text(request.text)
        return ActionResult(True, proposal.description)
