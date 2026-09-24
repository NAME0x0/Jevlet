"""The assistant's skill catalogue and the argument questions each skill needs.

Skill names and descriptions are the runtime criteria the model chooses between; argument
options are built from the live environment (installed apps, open windows, visible controls)
or copied from the command itself. The same builders produce training data and live queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .environment import KNOWN_FOLDERS, SETTINGS_PAGES, SHORTCUTS, WEBSITES
from .text import shortlist, span_candidates

NOT_APPLICABLE = "Not applicable"
SKILL_QUESTION = "Which action does the command ask for?"


@dataclass(frozen=True, slots=True)
class Slot:
    key: str
    question: str


SLOTS = {
    "app": Slot("app", "Which installed app does the command mean?"),
    "window": Slot("window", "Which open window does the command mean?"),
    "setting": Slot("setting", "Which Settings page does the command need?"),
    "website": Slot("website", "Which website does the command mean?"),
    "folder": Slot("folder", "Which folder does the command mean?"),
    "shortcut": Slot("shortcut", "Which keyboard shortcut does the command mean?"),
    "media": Slot("media", "Which playback action does the command mean?"),
    "volume": Slot("volume", "Which volume change does the command mean?"),
    "theme": Slot("theme", "Which color mode does the command mean?"),
    "text": Slot("text", "Which part of the command is the text to use?"),
    "service": Slot("service", "Which AI assistant should get the request?"),
}
MEDIA_ACTIONS = ("Play or pause", "Next track", "Previous track")
VOLUME_ACTIONS = ("Volume up", "Volume down", "Mute or unmute")
THEMES = ("Dark mode", "Light mode")
SERVICES = ("Claude", "ChatGPT", "Gemini")


@dataclass(frozen=True, slots=True)
class Skill:
    key: str
    name: str
    description: str
    slots: tuple[str, ...] = ()
    risk_floor: float = 0.0  # the gate never treats this skill as safer than this
    title: str = ""  # shown in the UI, formatted with argument values
    needs_screen: bool = False


SKILLS = (
    Skill("open_app", "Open an app", "Launch an installed application.", ("app",), title="Open {app}"),
    Skill("switch_window", "Switch window", "Bring an already open window to the front.", ("window",), title="Switch to {window}"),
    Skill("close_window", "Close window", "Close an open window; the app may ask to save work.", ("window",), 0.15, "Close {window}"),
    Skill("minimize_window", "Minimize window", "Minimize or hide an open window.", ("window",), title="Minimize {window}"),
    Skill("maximize_window", "Maximize window", "Maximize or enlarge an open window.", ("window",), title="Maximize {window}"),
    Skill("media", "Control playback", "Play, pause, or skip music and videos.", ("media",), title="{media}"),
    Skill("volume", "Change volume", "Turn the system volume up, down, or mute it.", ("volume",), title="{volume}"),
    Skill("settings", "Open settings", "Open a Windows Settings page.", ("setting",), title="Settings: {setting}"),
    Skill("theme", "Change theme", "Switch Windows between dark mode and light mode.", ("theme",), title="{theme}"),
    Skill("search", "Search the web", "Search the internet for something.", ("text",), title="Search the web for “{text}”"),
    Skill("website", "Open website", "Open a well-known website in the browser.", ("website",), title="Open {website}"),
    Skill("folder", "Open folder", "Open a folder in File Explorer.", ("folder",), title="Open {folder}"),
    Skill("type", "Type text", "Type the given words into the focused text field.", ("text",), 0.15, "Type “{text}”"),
    Skill("shortcut", "Press shortcut", "Press a keyboard shortcut such as copy, paste, undo, save, or new tab.", ("shortcut",), title="{shortcut}"),
    Skill("click", "Click on screen", "Click a button, tab, link, or menu item in the current window.", (), 0.1, "Click {control}", True),
    Skill("timer", "Start timer", "Start a countdown timer or a reminder.", (), title="Timer for {duration}"),
    Skill("screenshot", "Take screenshot", "Save a picture of the whole screen.", (), title="Take a screenshot"),
    Skill("lock", "Lock computer", "Lock the screen.", (), title="Lock the computer"),
    Skill("ask_ai", "Ask an AI", "Send a writing, explaining, analysis, or coding request to an AI assistant.", ("service", "text"), title="Ask {service}"),
    Skill("clarify", "Ask for clarification", "The command is unclear, not possible here, or needs the user's own judgment.", (), 0.5, "I need more detail"),
)  # fmt: skip
SKILL_BY_NAME = {skill.name: skill for skill in SKILLS}
SKILL_BY_KEY = {skill.key: skill for skill in SKILLS}


@dataclass(slots=True)
class Environment:
    """A snapshot of what exists right now; built live or sampled for training data."""

    apps: list[str] = field(default_factory=list)
    windows: list[str] = field(default_factory=list)  # "process: title", most recent first
    current_window: str = ""


def window_label(process: str, title: str) -> str:
    title = title if len(title) <= 60 else title[:57] + "..."
    return f"{process}: {title}" if process else title


def slot_options(slot: str, command: str, env: Environment) -> list[str]:
    """Runtime options for one argument, always ending with NOT_APPLICABLE."""
    if slot == "app":
        options = shortlist(command, env.apps, 8)
    elif slot == "window":
        others = [window for window in env.windows if window != env.current_window]
        options = ([f"Current window ({env.current_window})"] if env.current_window else [])
        options += shortlist(command, others, 7)
    # Short fixed catalogues are offered whole: "pair my headphones" shares no words with
    # "Bluetooth and devices", so a lexical shortlist would drop the answer.
    elif slot == "setting":
        options = list(SETTINGS_PAGES)
    elif slot == "website":
        options = list(WEBSITES)
    elif slot == "folder":
        options = list(KNOWN_FOLDERS)
    elif slot == "shortcut":
        options = list(SHORTCUTS)
    elif slot == "media":
        options = list(MEDIA_ACTIONS)
    elif slot == "volume":
        options = list(VOLUME_ACTIONS)
    elif slot == "theme":
        options = list(THEMES)
    elif slot == "service":
        options = list(SERVICES)
    elif slot == "text":
        options = span_candidates(command, 8)
    else:
        raise KeyError(slot)
    unique = list(dict.fromkeys(option for option in options if option))
    return unique + [NOT_APPLICABLE]
