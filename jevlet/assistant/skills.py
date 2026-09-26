"""The assistant's skill catalogue, loaded from ``shared/skills.json``.

The same JSON is embedded in the C# app, so the skills, argument questions, catalogues, and risk
floors used to generate training data are exactly the ones the product asks at runtime.
Argument options are built from the live environment (installed apps, open windows, stored
events and alarms, file search results) or copied from the command itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .text import shortlist, span_candidates

CATALOGUE_PATH = Path(__file__).resolve().parents[2] / "shared" / "skills.json"
_DATA = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))

FINGERPRINT_FIELD, FINGERPRINT_LINE, FINGERPRINT_ITEM = "\t", "\n", "\x1f"


def catalogue_fingerprint(data: dict, *, risk_question: str, state_format: str) -> str:
    """SHA-256 of everything a model reads from the catalogue: the questions, state format, and
    option names, in the order they are offered.

    ``Catalogue.Fingerprint`` in the C# app builds the same text, and the app refuses a model
    whose ``model.json`` carries a different fingerprint: a model trained on other questions or
    skill names would be answering blind. Execution details (URIs, key codes, icons, titles,
    descriptions, risk floors) are left out because the model never sees them. Catalogues older
    than v6's C# port lack ``risk_question``/``state_format``; the caller passes the values that
    training used.
    """
    rows = [
        ["version", str(data["version"])],
        ["skill_question", data["skill_question"]],
        ["risk_question", data.get("risk_question", risk_question)],
        ["state_format", data.get("state_format", state_format)],
        ["not_applicable", data["not_applicable"]],
    ]
    for skill in data["skills"]:
        slots = ",".join(skill.get("slots", ()))
        optional = ",".join(skill.get("optional_slots", ()))
        rows.append(["skill", skill["key"], skill["name"], slots, optional])
    for key in sorted(data["slots"]):
        rows.append(["slot", key, data["slots"][key]])
    for key in sorted(data["catalogues"]):
        values = data["catalogues"][key]
        names = list(values) if isinstance(values, dict) else [str(value) for value in values]
        rows.append(["catalogue", key, FINGERPRINT_ITEM.join(names)])
    text = FINGERPRINT_LINE.join(FINGERPRINT_FIELD.join(row) for row in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


NOT_APPLICABLE: str = _DATA["not_applicable"]
SKILL_QUESTION: str = _DATA["skill_question"]
CATALOGUES: dict = _DATA["catalogues"]


@dataclass(frozen=True, slots=True)
class Slot:
    key: str
    question: str


SLOTS = {key: Slot(key, question) for key, question in _DATA["slots"].items()}
MEDIA_ACTIONS = tuple(CATALOGUES["media"])
VOLUME_ACTIONS = tuple(CATALOGUES["volume"])
THEMES = tuple(CATALOGUES["theme"])
SERVICES = tuple(CATALOGUES["service"])
FIXED_CHOICES = {
    name: tuple(CATALOGUES[name])
    for name in (
        "media", "volume", "theme", "service", "power", "radio", "brightness", "stopwatch",
        "timer_action", "alarm_action",
    )
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class Skill:
    key: str
    name: str
    description: str
    slots: tuple[str, ...] = ()
    risk_floor: float = 0.0  # the gate never treats this skill as safer than this
    title: str = ""  # shown in the UI, formatted with argument values
    needs_screen: bool = False
    optional_slots: tuple[str, ...] = ()
    time: str = ""  # "", "duration", "required", or "optional"
    icon: str = ""  # Segoe Fluent Icons code point used by the C# app


SKILLS = tuple(
    Skill(
        key=row["key"],
        name=row["name"],
        description=row["description"],
        slots=tuple(row.get("slots", ())),
        risk_floor=float(row.get("risk_floor", 0.0)),
        title=row.get("title", ""),
        needs_screen=bool(row.get("needs_screen", False)),
        optional_slots=tuple(row.get("optional_slots", ())),
        time=row.get("time") or "",
        icon=row.get("icon", ""),
    )
    for row in _DATA["skills"]
)
SKILL_BY_NAME = {skill.name: skill for skill in SKILLS}
SKILL_BY_KEY = {skill.key: skill for skill in SKILLS}


@dataclass(slots=True)
class Environment:
    """A snapshot of what exists right now; built live or sampled for training data."""

    apps: list[str] = field(default_factory=list)
    windows: list[str] = field(default_factory=list)  # "process: title", most recent first
    current_window: str = ""
    events: list[str] = field(default_factory=list)  # "title · when"
    alarms: list[str] = field(default_factory=list)  # "07:30 · label"
    todos: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)  # file names from a search
    reminders: list[str] = field(default_factory=list)  # "task · when"


def window_label(process: str, title: str) -> str:
    title = title if len(title) <= 60 else title[:57] + "..."
    return f"{process}: {title}" if process else title


def slot_options(slot: str, command: str, env: Environment) -> list[str]:
    """Runtime options for one argument, always ending with NOT_APPLICABLE."""
    if slot == "app":
        options = shortlist(command, env.apps, 8)
    elif slot == "window":
        others = [window for window in env.windows if window != env.current_window]
        options = [f"Current window ({env.current_window})"] if env.current_window else []
        options += shortlist(command, others, 7)
    elif slot in {"event", "alarm", "todo", "file", "reminder"}:
        pool = {
            "event": env.events,
            "alarm": env.alarms,
            "todo": env.todos,
            "file": env.files,
            "reminder": env.reminders,
        }
        options = shortlist(command, pool[slot], 8)
    # Short fixed catalogues are offered whole: "pair my headphones" shares no words with
    # "Bluetooth and devices", so a lexical shortlist would drop the answer.
    elif slot in {"setting", "website", "folder", "shortcut"}:
        options = list(CATALOGUES[slot])
    elif slot in FIXED_CHOICES:
        options = list(FIXED_CHOICES[slot])
    elif slot == "text":
        options = span_candidates(command)
    else:
        raise KeyError(slot)
    unique = list(dict.fromkeys(option for option in options if option))
    return unique + [NOT_APPLICABLE]
