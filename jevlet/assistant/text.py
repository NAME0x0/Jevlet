"""Cheap, deterministic text helpers around the model.

* ``shortlist``: first stage of Jev's two-stage choice. Lexical and alias scoring narrows a
  large runtime list (hundreds of apps) to a few options the model compares listwise.
* ``span_candidates``: the model never generates text. Free-text arguments (a search query,
  text to type) are chosen from spans of the user's own command.
* ``parse_duration``: numbers and units are parsed, not predicted (Jev is weak at arithmetic).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

ALIASES = {
    "music": ("spotify", "media player", "groove", "apple music"),
    "songs": ("spotify",),
    "browser": ("edge", "chrome", "firefox", "brave"),
    "web": ("edge", "chrome", "firefox"),
    "internet": ("edge", "chrome", "firefox"),
    "mail": ("outlook", "mail", "thunderbird"),
    "email": ("outlook", "mail", "thunderbird"),
    "inbox": ("outlook", "mail"),
    "code": ("visual studio code", "pycharm", "cursor"),
    "editor": ("visual studio code", "notepad", "notepad++"),
    "ide": ("visual studio code", "pycharm", "visual studio"),
    "notes": ("onenote", "notepad", "obsidian", "sticky notes"),
    "terminal": ("terminal", "windows terminal", "powershell", "command prompt"),
    "shell": ("terminal", "powershell", "command prompt"),
    "files": ("file explorer",),
    "folders": ("file explorer",),
    "calc": ("calculator",),
    "spreadsheet": ("excel",),
    "slides": ("powerpoint",),
    "presentation": ("powerpoint",),
    "document": ("word",),
    "chat": ("teams", "slack", "whatsapp", "discord"),
    "meeting": ("teams", "zoom"),
    "video call": ("teams", "zoom"),
    "photos": ("photos",),
    "pictures": ("photos",),
    "settings": ("settings",),
    "control panel": ("control panel",),
    "store": ("microsoft store",),
    "paint": ("paint",),
    "camera": ("camera",),
    "clock": ("clock",),
    "alarm": ("clock",),
    "timer": ("clock",),
    "task manager": ("task manager",),
}

WORD = re.compile(r"[a-z0-9+#.]+")
UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
}  # fmt: skip
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
    "half": 0.5, "quarter": 0.25,
}  # fmt: skip
TRIGGERS = (
    "search the web for", "search for", "look up", "google", "search", "find",
    "type out", "type", "write", "say", "enter", "reply with", "named", "called",
    "titled", "about", "for",
)  # fmt: skip


def words(text: str) -> list[str]:
    return WORD.findall(text.casefold())


def _trigrams(text: str) -> set[str]:
    padded = f"  {text.casefold()} "
    return {padded[index : index + 3] for index in range(len(padded) - 2)}


def similarity(query: str, name: str) -> float:
    """Blend of word overlap, character trigrams, and alias hits; 0 means unrelated."""
    query_words, name_words = set(words(query)), set(words(name))
    if not name_words:
        return 0.0
    overlap = len(query_words & name_words) / len(name_words)
    grams_query, grams_name = _trigrams(query), _trigrams(name)
    trigram = len(grams_query & grams_name) / max(len(grams_name), 1)
    alias = 0.0
    lowered_name, lowered_query = name.casefold(), query.casefold()
    for trigger, targets in ALIASES.items():
        if re.search(rf"\b{re.escape(trigger)}\b", lowered_query) and any(
            target in lowered_name for target in targets
        ):
            alias = 0.8
            break
    return max(overlap, 0.6 * trigram, alias)


def shortlist(query: str, names: Iterable[str], limit: int = 8) -> list[str]:
    """Top candidates by similarity; ties keep the caller's order (e.g. recency)."""
    scored = [(similarity(query, name), index, name) for index, name in enumerate(names)]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, _, name in scored[:limit]]


COURTESY = re.compile(
    r"^(?:(?:hey|hi|ok|okay)\s+jevlet[,:]?\s*|jevlet[,:]\s*|please\s+|can you\s+|could you\s+|"
    r"would you\s+|i want to\s+|i need to\s+|i'd like to\s+|quickly\s+|just\s+)+",
    re.I,
)
DELEGATE = re.compile(r"\b(?:ask|have|get|let|use|tell|make)\s+\w+(?:\s+to)?\s+|\b\w+,\s+", re.I)


def strip_courtesy(command: str) -> str:
    """'hey jevlet, can you open X' -> 'open X' (the words that carry the request)."""
    return COURTESY.sub("", command.strip()).strip()


def span_candidates(command: str, limit: int = 8) -> list[str]:
    """Plausible free-text arguments, all copied verbatim from the command."""
    text = strip_courtesy(command).rstrip(". ")
    candidates: list[str] = []
    candidates += re.findall(r"[\"“']([^\"”']{1,200})[\"”']", text)
    # "ask Claude to X", "have Gemini X", "ChatGPT, X": the request follows the assistant.
    candidates += [text[match.end() :] for match in DELEGATE.finditer(text)][:2]
    lowered = text.casefold()
    for trigger in TRIGGERS:
        for match in re.finditer(rf"\b{re.escape(trigger)}\b\s+", lowered):
            span = text[match.end() :].strip()
            for stop in (
                " and then ", " then ", " in ", " on ", " into ", " using ", " online",
                " for me", " please", " here", " right now", " quickly",
            ):  # fmt: skip
                cut = span.casefold().find(stop)
                if cut > 0:
                    candidates.append(span[:cut].strip())
            candidates.append(span)
    tokens = text.split()
    candidates.append(text)
    if len(tokens) > 1:
        candidates.append(" ".join(tokens[1:]))
    for size in (1, 2, 3, 4):
        if len(tokens) > size:
            candidates.append(" ".join(tokens[-size:]))
    unique: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip(" ,:;\"'“”")
        if cleaned and cleaned.casefold() not in {item.casefold() for item in unique}:
            unique.append(cleaned)
    return unique[:limit]


def parse_duration(command: str) -> int | None:
    """Seconds for phrases like '25 minutes', 'an hour and a half', '90s'; None if absent."""
    text = command.casefold().replace("-", " ")
    total = 0.0
    found = False
    for number, unit in re.findall(
        r"(\d+(?:\.\d+)?|[a-z]+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b", text
    ):
        value = float(number) if number[0].isdigit() else NUMBER_WORDS.get(number)
        if value is None:
            continue
        total += value * UNITS[unit]
        found = True
    if "and a half" in text and found:
        unit = 3600 if "hour" in text else 60
        total += 0.5 * unit
    return int(total) if found and total > 0 else None


def contains_any(text: str, phrases: Sequence[str]) -> bool:
    lowered = text.casefold()
    return any(phrase in lowered for phrase in phrases)
