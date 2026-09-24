"""Command -> plan with calibrated confidence, risk, and a gate; then execution.

Phase 1 asks the skill and risk questions together; phase 2 asks only the chosen skill's
argument questions. Both are parallel packed passes over the same state (the command plus the
active window). A multi-step command ("open Spotify then play") is split on "then"; later steps
are re-planned against the live screen right before they run.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from jevlet.benchmarks import RISK_QUESTION, daily_state
from jevlet.system_one import ChoiceQuestion, NoulQuestion, SystemOne

from . import actions
from .environment import (
    KNOWN_FOLDERS,
    SETTINGS_PAGES,
    SHORTCUTS,
    WEBSITES,
    App,
    Window,
    installed_apps,
    open_windows,
)
from .skills import (
    NOT_APPLICABLE,
    SKILL_BY_NAME,
    SKILL_QUESTION,
    SKILLS,
    SLOTS,
    Environment,
    Skill,
    slot_options,
    window_label,
)
from .text import parse_duration

SAFE_TO_RUN = 0.9  # Jev's bar for destructive operations: P(safe) must reach 0.9
CONFIDENT = 0.75  # skill and argument confidence needed to run on a single Enter


@dataclass(slots=True)
class Context:
    """What was on screen when the palette opened (the palette itself is excluded)."""

    current: Window | None
    windows: list[Window]
    apps: list[App]

    @classmethod
    def capture(cls, exclude_handles: tuple[int, ...] = ()) -> Context:
        windows = open_windows(exclude_handles)
        return cls(windows[0] if windows else None, windows, installed_apps())

    def environment(self) -> Environment:
        return Environment(
            apps=[app.name for app in self.apps],
            windows=[window_label(w.process, w.title) for w in self.windows],
            current_window=window_label(self.current.process, self.current.title)
            if self.current
            else "",
        )


@dataclass(slots=True)
class Plan:
    command: str
    skill: Skill
    confidence: float
    risk: float
    args: dict[str, str] = field(default_factory=dict)
    arg_confidence: dict[str, float] = field(default_factory=dict)
    alternatives: list[tuple[Skill, float]] = field(default_factory=list)
    duration: int | None = None
    control: object | None = None
    latency_ms: float = 0.0
    missing: str = ""

    @property
    def title(self) -> str:
        values = {slot: _display(slot, value) for slot, value in self.args.items()}
        values["duration"] = _format_duration(self.duration)
        if self.control is not None:
            values["control"] = f"“{self.control.name or self.control.automation_id}”"
        try:
            return self.skill.title.format(**values)
        except KeyError:
            return self.skill.name

    @property
    def gate(self) -> str:
        """run: one Enter executes. confirm: a second Enter is required. clarify: cannot run."""
        if self.skill.key == "clarify" or self.missing:
            return "clarify"
        weakest = min([self.confidence, *self.arg_confidence.values()])
        if 1.0 - self.risk >= SAFE_TO_RUN and weakest >= CONFIDENT:
            return "run"
        return "confirm"


def _display(slot: str, value: str) -> str:
    """Human wording for the UI; the model's options keep the process name for precision."""
    if slot != "window":
        return value
    if value.startswith("Current window"):
        return "this window"
    return value.split(": ", 1)[1] if ": " in value else value


def _format_duration(seconds: int | None) -> str:
    if not seconds:
        return "?"
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    parts = [
        f"{hours} h" if hours else "",
        f"{minute} min" if minute else "",
        f"{second} s" if second else "",
    ]
    return " ".join(part for part in parts if part)


def split_steps(command: str) -> list[str]:
    steps = re.split(
        r"\s*(?:,\s*)?\b(?:and then|then|after that)\b\s*", command.strip(), flags=re.I
    )
    return [step.strip(" ,.") for step in steps if step.strip(" ,.")]


class Planner:
    def __init__(self, engine: SystemOne) -> None:
        self.engine = engine
        # Many argument options can exceed training-time packing limits; positions still fit.
        config = getattr(self.engine.collator, "config", None)
        if config is not None:
            config.max_packed_len = max(config.max_packed_len, 2048)

    def _state(self, command: str, context: Context) -> str:
        current = context.current
        return daily_state(command, window_label(current.process, current.title) if current else "")

    def plan(self, command: str, context: Context) -> Plan:
        started = time.perf_counter()
        state = self._state(command, context)
        first = self.engine.evaluate(
            state,
            {
                "skill": ChoiceQuestion(SKILL_QUESTION, {s.name: s.description for s in SKILLS}),
                "risk": NoulQuestion(RISK_QUESTION),
            },
        )
        choice = first["skill"]
        ranked = sorted(
            zip(choice.options, choice.probabilities, strict=True), key=lambda pair: -pair[1]
        )
        skill = SKILL_BY_NAME[ranked[0][0]]
        plan = self.plan_for(skill, command, context, ranked[0][1], first["risk"].probability_true)
        plan.alternatives = [(SKILL_BY_NAME[name], p) for name, p in ranked[1:4]]
        plan.latency_ms = (time.perf_counter() - started) * 1000
        return plan

    def plan_for(
        self, skill: Skill, command: str, context: Context, confidence: float, risk: float
    ) -> Plan:
        """Fill one skill's arguments (used for the top skill and for chosen alternatives)."""
        plan = Plan(command, skill, confidence, max(risk, skill.risk_floor))
        env = context.environment()
        questions = {}
        for slot in skill.slots:
            options = slot_options(slot, command, env)
            if len(options) > 1:
                questions[slot] = ChoiceQuestion(SLOTS[slot].question, options)
            else:
                plan.missing = slot
        if questions:
            answers = self.engine.evaluate(self._state(command, context), questions)
            for slot, answer in answers.items():
                plan.args[slot] = answer.selected
                plan.arg_confidence[slot] = answer.confidence
                if answer.selected == NOT_APPLICABLE:
                    plan.missing = slot
        if skill.key == "timer":
            plan.duration = parse_duration(command)
            if not plan.duration:
                plan.missing = "duration"
        if skill.needs_screen:
            self._ground(plan, context)
        return plan

    def _ground(self, plan: Plan, context: Context) -> None:
        from jevlet.desktop.uia import FastUIA
        from jevlet.grounding import NONE_OF_THESE, ground

        if context.current is None:
            plan.missing = "control"
            return
        controls = FastUIA().read(context.current.handle)
        target, choice, risk = ground(self.engine, plan.command, context.current.title, controls)
        plan.control = target
        plan.arg_confidence["control"] = choice.confidence
        plan.risk = max(plan.risk, risk)
        if target is None or choice.selected == NONE_OF_THESE:
            plan.missing = "control"


def _window_for(label: str, context: Context) -> Window | None:
    if label.startswith("Current window") and context.current:
        return context.current
    for window in context.windows:
        if window_label(window.process, window.title) == label:
            return window
    return None


def execute(plan: Plan, context: Context) -> actions.Outcome:
    key, args = plan.skill.key, plan.args
    previous = context.current.handle if context.current else None
    if key == "open_app":
        app = next(app for app in context.apps if app.name == args["app"])
        return actions.launch_app(app.app_id, app.name)
    if key in {"switch_window", "close_window", "minimize_window", "maximize_window"}:
        window = _window_for(args["window"], context)
        if window is None:
            return actions.Outcome(False, "That window is no longer open")
        if key == "switch_window":
            return actions.switch_to(window.handle, window.title, previous)
        return actions.window_state(window.handle, window.title, key.split("_")[0])
    if key == "media":
        return actions.media(args["media"])
    if key == "volume":
        return actions.volume(args["volume"])
    if key == "settings":
        return actions.open_target(SETTINGS_PAGES[args["setting"]], args["setting"])
    if key == "theme":
        return actions.theme(args["theme"])
    if key == "search":
        return actions.web_search(args["text"])
    if key == "website":
        return actions.open_target(WEBSITES[args["website"]], args["website"])
    if key == "folder":
        return actions.open_target(KNOWN_FOLDERS[args["folder"]], args["folder"])
    if key == "type":
        return actions.type_text(previous, args["text"])
    if key == "shortcut":
        return actions.shortcut(args["shortcut"], SHORTCUTS[args["shortcut"]], previous)
    if key == "click":
        return actions.click_control(previous, plan.control)
    if key == "screenshot":
        return actions.screenshot()
    if key == "lock":
        return actions.lock()
    if key == "ask_ai":
        return actions.ask_ai(args["service"], args["text"])
    if key == "timer":
        return actions.Outcome(True, f"Timer set for {_format_duration(plan.duration)}")
    return actions.Outcome(False, "I need more detail to do that")
