"""Cheap, deterministic text helpers around the model.

* ``shortlist``: first stage of Jev's two-stage choice. Lexical and alias scoring narrows a
  large runtime list (hundreds of apps) to a few options the model compares listwise.
* ``span_candidates``: the model never generates text. Free-text arguments (a search query,
  text to type) are chosen from spans of the user's own command.
* ``parse_duration``: numbers and units are parsed, not predicted (Jev is weak at arithmetic).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

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
# Span rules live in shared/text_rules.json so the C# app applies the same lists in the same
# order. A span is also offered cut at the first of each stop, so "lunch with omar next
# tuesday at 1" yields "lunch with omar".
RULES = json.loads(
    (Path(__file__).resolve().parents[2] / "shared" / "text_rules.json").read_text(encoding="utf-8")
)
TRIGGERS: tuple[str, ...] = tuple(RULES["triggers"])
TRIGGER_PATTERNS = tuple(re.compile(rf"\b{re.escape(trigger)}\b[:,]?\s+") for trigger in TRIGGERS)
STOPS: tuple[str, ...] = tuple(RULES["stops"])
LABEL_BEFORE_NOUN = re.compile(RULES["label_before_noun"], re.I)


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


COURTESY = re.compile(RULES["courtesy"], re.I)
DELEGATE = re.compile(RULES["delegate"], re.I)
TEXT_CANDIDATES: int = RULES["text_candidates"]


def strip_courtesy(command: str) -> str:
    """'hey jevlet, can you open X' -> 'open X' (the words that carry the request).

    Trailing courtesy ("thanks") is left in place and handled by span cuts instead: "type
    thank you" must keep its argument.
    """
    return COURTESY.sub("", command.strip()).strip()


def _with_cuts(span: str) -> list[str]:
    lowered = span.casefold()
    cuts = [span[: lowered.find(stop)].strip() for stop in STOPS if lowered.find(stop) > 0]
    return cuts + [span]


PREPOSITION = re.compile(RULES["preposition"], re.I)
PHRASE_END = re.compile(RULES["phrase_end"], re.I)
PROPER_RUN = re.compile(RULES["proper_run_case_sensitive"])


def _place_phrases(text: str) -> list[str]:
    """Short noun phrases after a preposition ("to madison square garden by 9") and runs of
    capitalized words not at the start ("the North East England forecast")."""
    phrases = []
    for match in PREPOSITION.finditer(text):
        rest = text[match.end() :]
        end = PHRASE_END.search(rest)
        phrase = (rest[: end.start()] if end else rest).strip()
        if 0 < len(phrase.split()) <= 5:
            phrases.append(phrase)
    for match in PROPER_RUN.finditer(text):
        run = re.sub(r"'s$", "", match.group(0).rstrip(".,'"))
        if match.start() > 0 and run not in {"I", "I'm", "I'll", "I'd"}:
            phrases.append(run)
    return phrases


def span_candidates(command: str, limit: int = TEXT_CANDIDATES) -> list[str]:
    """Plausible free-text arguments, all copied verbatim from the command."""
    # Terminal punctuation is never part of an unquoted argument ("weather in cairo?").
    text = strip_courtesy(command).rstrip(" .?!")
    candidates: list[str] = []
    candidates += re.findall(r"[\"“']([^\"”']{1,200})[\"”']", text)
    # "ask Claude to X", "have Gemini X", "ChatGPT, X": the request follows the assistant.
    candidates += [text[match.end() :] for match in DELEGATE.finditer(text)][:2]
    # "new task: X", "new alarm - X", but not "11:20"
    labelled = re.split(r":\s+|\s+-\s+", text, maxsplit=1)
    if len(labelled) == 2:
        candidates += _with_cuts(labelled[1].strip())
    # "set a wakeup alarm", "create a lunch time reminder": the label sits before the noun.
    candidates += [match.group(1) for match in LABEL_BEFORE_NOUN.finditer(text)][:2]
    # Triggers also run on the command with its time phrases removed, so "ping me this sunday
    # at 1 to submit the claim" still yields "submit the claim" through "me to".
    untimed = " ".join(TIME_EXPRESSION.sub(" ", text).split())
    for pattern in TRIGGER_PATTERNS:
        for source in (text, untimed) if untimed != text else (text,):
            for match in pattern.finditer(source.casefold()):
                candidates += _with_cuts(source[match.end() :].strip())
    candidates += _place_phrases(text)
    tokens = text.split()
    candidates += _with_cuts(text)
    if len(tokens) > 1:
        candidates += _with_cuts(" ".join(tokens[1:]))
    for size in (1, 2, 3, 4):
        if len(tokens) > size:
            candidates.append(" ".join(tokens[-size:]))
    unique: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip(" ,:;\"'“”")
        if cleaned and cleaned.casefold() not in {item.casefold() for item in unique}:
            unique.append(cleaned)
    return unique[:limit]


TIME_EXPRESSION = re.compile(RULES["time_expression"], re.I)


def has_time(command: str) -> bool:
    """Whether a command names a time or date (the C# app parses it fully)."""
    return bool(TIME_EXPRESSION.search(command))


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
