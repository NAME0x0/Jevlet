"""Grounding data from the user's own apps: captured control trees + a generic task lexicon.

``capture`` reads the actionable controls of every open window (read-only; nothing is clicked)
and merges them per process into ``data/inventories``. ``inventory_apps`` turns each captured
app into a grounding ``App`` whose targets are the controls a generic lexicon recognizes
("Close Tab" -> "close this tab", "Send" -> "send it", risky). Everything stays local.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .grounding import actionable
from .grounding_synthetic import App

INVENTORIES = Path(__file__).resolve().parent.parent / "data" / "inventories"

# (pattern on the control name, task phrasings, risky)
LEXICON = (
    (r"^close tab$", ("close this tab", "shut the current tab"), False),
    (r"^new tab$|^open a new tab$", ("open a new tab", "give me another tab"), False),
    (r"^(back|go back)$", ("go back", "previous page"), False),
    (r"^forward$", ("go forward",), False),
    (r"^(refresh|reload)$", ("refresh this", "reload it"), False),
    (r"^minimi[sz]e$", ("minimize this window", "hide this"), False),
    (r"^maximi[sz]e$", ("maximize this window", "make this window bigger"), False),
    (r"^(search|search box|find)$", ("search in here", "find something in this app"), False),
    (r"settings$|^preferences$|^options$", ("open the settings here", "show preferences"), False),
    (r"^(send|send now|post)$", ("send it", "send the message"), True),
    (r"^delete$|^remove$|^trash$", ("delete this", "remove it"), True),
    (r"^share$", ("share this",), True),
    (r"^save$", ("save this", "save my changes"), False),
    (r"^(copy|paste|cut|undo|redo)$", None, False),  # phrased from the control name itself
    (r"^(play|pause|play/pause)$", ("pause", "play"), False),
    (r"^(next|next track|skip)$", ("next one", "skip ahead"), False),
    (r"^(previous|previous track)$", ("previous one", "go back a track"), False),
    (r"^mute$|^unmute$", ("mute", "toggle mute"), False),
    (r"^(attach|attach file|add attachment)$", ("attach a file",), False),
    (r"^(reply|reply all|forward)$", None, False),
    (r"^(download)$", ("download it",), False),
    (r"^(upload|upload a file)$", ("upload a file",), True),
    (r"^(home)$", ("go to the home page",), False),
    (r"^(help)$", ("get help",), False),
    (r"^(more options|more actions|more)$", ("show more options",), False),
    (r"^(zoom in)$", ("zoom in",), False),
    (r"^(zoom out)$", ("zoom out",), False),
    (r"^(print)$", ("print this",), False),
    (r"^(full ?screen)$", ("go full screen",), False),
)


def _tasks_for(name: str) -> tuple[tuple[str, ...], bool] | None:
    # Task text goes through str.format later, so braces from app UIs must be neutralized.
    lowered = name.casefold().strip().replace("{", "(").replace("}", ")")
    for pattern, phrasings, risky in LEXICON:
        if re.search(pattern, lowered):
            return (phrasings or (f"{lowered} this", lowered)), risky
    return None


def capture(output: Path = INVENTORIES) -> dict[str, int]:
    """Merge the actionable controls of all open windows into per-process inventories."""
    from .assistant.environment import open_windows
    from .desktop.uia import FastUIA

    reader = FastUIA()
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for window in open_windows():
        if not window.process:
            continue
        try:
            controls = actionable(reader.read(window.handle))
        except Exception:  # noqa: BLE001 - some windows refuse UIA; skip them
            continue
        path = output / f"{window.process.casefold()}.json"
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        titles = sorted(set(stored.get("titles", [])) | {window.title[:80]})[-10:]
        known = {(c[0], c[1]) for c in stored.get("controls", [])}
        known |= {(c.control_type, c.name) for c in controls if c.name.strip()}
        path.write_text(
            json.dumps(
                {"process": window.process, "titles": titles, "controls": sorted(known)}, indent=1
            ),
            encoding="utf-8",
        )
        counts[window.process] = len(known)
    return counts


def inventory_apps(directory: Path = INVENTORIES, minimum_targets: int = 2) -> tuple[App, ...]:
    apps = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        controls = tuple((kind, name) for kind, name in data["controls"] if len(name) <= 60)
        tasks, risky = {}, set()
        for _, name in controls:
            found = _tasks_for(name)
            if found and name not in tasks:
                tasks[name] = found[0]
                if found[1]:
                    risky.add(name)
        # Control names repeated with different types would make the target ambiguous.
        names = [name for _, name in controls]
        controls = tuple(c for c in controls if names.count(c[1]) == 1 or c[1] not in tasks)
        tasks = {name: phr for name, phr in tasks.items() if names.count(name) == 1}
        if len(tasks) >= minimum_targets:
            titles = tuple(title.replace("{", "(").replace("}", ")") for title in data["titles"])
            apps.append(
                App(
                    f"live:{data['process']}",
                    titles,
                    controls,
                    tasks,
                    frozenset(risky) & set(tasks),
                )
            )
    return tuple(apps)
