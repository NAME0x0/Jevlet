"""Screen grounding: choose which visible UI Automation control accomplishes a task.

The options are the controls UI Automation reports for the foreground window, so they are
runtime-defined text options exactly like Jev's criteria. The same formatting is used for
generated training data and for live inference, so the model sees one representation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

NONE_OF_THESE = "None of these controls (the task needs another app or window)"
GROUNDING_QUESTION = "Which on-screen control should be used for this task?"
ACTIONABLE_TYPES = frozenset(
    {
        "Button",
        "SplitButton",
        "Edit",
        "ComboBox",
        "MenuItem",
        "ListItem",
        "TreeItem",
        "TabItem",
        "CheckBox",
        "RadioButton",
        "Hyperlink",
        "Slider",
        "DataItem",
    }
)
MAX_CONTROLS = 254  # Jev caps a choice at 255 options; one slot is NONE_OF_THESE.


def control_text(control_type: str, name: str, automation_id: str = "") -> str:
    label = name.strip() or (f"[{automation_id.strip()}]" if automation_id.strip() else "")
    return f"{control_type}: {label}"


def actionable(controls: Iterable[object]) -> list[object]:
    """Keep enabled, labeled, interactive controls; drop duplicates by displayed text."""
    kept: list[object] = []
    seen: set[str] = set()
    for control in controls:
        kind = getattr(control, "control_type", "")
        name = getattr(control, "name", "") or ""
        automation_id = getattr(control, "automation_id", "") or ""
        if kind not in ACTIONABLE_TYPES or not getattr(control, "enabled", True):
            continue
        if not (name.strip() or automation_id.strip()):
            continue
        text = control_text(kind, name, automation_id)
        if text in seen:
            continue
        seen.add(text)
        kept.append(control)
    return kept[:MAX_CONTROLS]


def grounding_options(controls: Sequence[object]) -> list[str]:
    options = [
        control_text(
            getattr(control, "control_type", ""),
            getattr(control, "name", "") or "",
            getattr(control, "automation_id", "") or "",
        )
        for control in controls
    ]
    return options + [NONE_OF_THESE]


def ground(engine, task: str, window_title: str, controls: Sequence[object]):
    """Ask a SystemOne engine for the target control and the task's risk in one pass.

    Returns ``(control or None, choice, risk_probability)``; ``None`` means no visible control
    fits and the task should be routed elsewhere.
    """
    from .benchmarks import RISK_QUESTION, daily_state
    from .system_one import ChoiceQuestion, NoulQuestion

    candidates = actionable(controls)
    options = grounding_options(candidates)
    answers = engine.evaluate(
        daily_state(task, window_title),
        {
            "control": ChoiceQuestion(GROUNDING_QUESTION, options),
            "risk": NoulQuestion(RISK_QUESTION),
        },
    )
    choice = answers["control"]
    index = options.index(choice.selected)
    target = candidates[index] if index < len(candidates) else None
    return target, choice, answers["risk"].probability_true
