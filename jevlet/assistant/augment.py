"""Surface variation for generated commands: how people actually type into a launcher.

Covers courtesy and filler words, text-speak, contractions, dropped apostrophes and articles,
British spelling, typos, casing, and punctuation. The gold text argument (a verbatim span of
the command) is protected from token edits; casing is applied to it as well so it stays a
verbatim span. Callers re-check that every gold argument is still offered and fall back to the
plain command when one is not.
"""

from __future__ import annotations

import random
import re

QUESTION_START = re.compile(
    r"^(?:what|what's|whats|how|how's|is|are|am|do|does|did|which|when|where|who|will|any|can|could|should)\b",
    re.I,
)
PREFIXES_ANY = (
    "hey jevlet, ", "jevlet, ", "ok jevlet ", "hey, ", "ok ", "so ", "um ", "uh, ", "erm ", "alright, ",
    "right, ", "quick one: ", "one more thing, ", "yo ", "hmm ",
)  # fmt: skip
PREFIXES_IMPERATIVE = (
    "please ", "pls ", "plz ", "can you ", "can u ", "could you ", "would you ", "i want to ", "i need to ",
    "i'd like to ", "quickly ", "just ", "go ahead and ", "can you please ", "could you please ",
    "before I forget, ", "when you get a sec, ", "real quick, ",
)  # fmt: skip
# "can you ..." only reads naturally before a verb: "can you set an alarm", not "can you alarm 7".
VERBS = frozenset("""
open launch start run close quit exit switch go show set add put remind delete cancel remove turn
play pause skip mute unmute search google look find type write enter take make move reschedule push
mark tick check note jot calculate work lock restart reboot shut sign log draft compose email ask have
get use tell let snooze stop dismiss silence resume continue reset clear bring pull jump hide minimize
maximize enlarge dim lower increase raise enable disable connect pair change navigate route book
schedule block create pencil complete cross save copy paste undo redo refresh reload zoom print select
reopen click press hit tap capture screenshot give extend freeze unpause delay shift bump call text
order buy list read load boot fire flip focus kill expand collapse tuck blast throw queue listen hear
time count begin end disconnect reconnect brighten empty format wipe erase transfer send post share
uninstall accept keep scrap wake convert compute translate summarize explain fix plan review rewrite
outline brainstorm describe fill reply say ping nudge
""".split())  # fmt: skip
SUFFIXES = (" please", " pls", " thanks", " thank you", " thx", " for me", " asap", " real quick", " cheers", " when you can")
TEXT_SPEAK = {
    "please": ("pls", "plz"), "you": ("u",), "your": ("ur",), "tomorrow": ("tmrw", "tmr", "2moro"),
    "tonight": ("tonite",), "minutes": ("mins",), "minute": ("min",), "hours": ("hrs",), "seconds": ("secs",),
    "thanks": ("thx",), "message": ("msg",), "right now": ("rn",), "okay": ("ok",), "because": ("cos", "bc"),
    "appointment": ("appt",), "calendar": ("cal",), "with": ("w/",), "something": ("smth",),
}  # fmt: skip
CONTRACTIONS = {
    "what is": "what's", "i am": "i'm", "do not": "don't", "cannot": "can't", "it is": "it's",
    "i would": "i'd", "let us": "let's", "is not": "isn't", "i will": "i'll", "that is": "that's",
    "how is": "how's", "where is": "where's", "there is": "there's", "will not": "won't",
    "did not": "didn't", "does not": "doesn't", "i have": "i've",
}  # fmt: skip
EXPANSIONS = {short: long for long, short in CONTRACTIONS.items()}
BRITISH = {
    "color": "colour", "favorite": "favourite", "center": "centre", "organize": "organise",
    "kilometers": "kilometres", "meters": "metres", "theater": "theatre", "gray": "grey",
    "practice": "practise", "canceled": "cancelled", "traveling": "travelling",
}  # fmt: skip
FILLERS = ("um", "uh", "like", "erm", "so", "ok")
KEYBOARD = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok",
    "o": "ipl", "p": "ol", "a": "qsz", "s": "adwx", "d": "sfec", "f": "dgrv", "g": "fhtb", "h": "gjyn",
    "j": "hkum", "k": "jli", "l": "ko", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}  # fmt: skip


def _replace_words(text: str, table: dict[str, object], rng: random.Random, rate: float) -> str:
    for source, target in table.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.I)
        if pattern.search(text) and rng.random() < rate:
            choice = rng.choice(target) if isinstance(target, tuple) else target
            text = pattern.sub(choice, text, count=1)
    return text


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    index = rng.randrange(1, len(word) - 1)
    kind = rng.random()
    if kind < 0.3:
        return word[:index] + word[index + 1 :]
    if kind < 0.55:
        return word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    if kind < 0.75:
        return word[:index] + word[index] + word[index:]
    neighbours = KEYBOARD.get(word[index].lower())
    return word[:index] + rng.choice(neighbours) + word[index + 1 :] if neighbours else word


def _token_noise(text: str, rng: random.Random, ops: set[str]) -> str:
    if not text.strip():
        return text
    if "contract" in ops:
        text = _replace_words(text, CONTRACTIONS, rng, 0.7)
    if "expand" in ops:
        text = _replace_words(text, EXPANSIONS, rng, 0.7)
    if "apostrophe" in ops:
        text = re.sub(r"\b(\w+)'(s|m|t|ll|d|re|ve)\b", r"\1\2", text)
    if "textspeak" in ops:
        text = _replace_words(text, TEXT_SPEAK, rng, 0.6)
    if "british" in ops:
        text = _replace_words(text, BRITISH, rng, 0.9)
    if "articles" in ops:
        text = re.sub(r"\b(?:the|a|an|my) (?=\w)", lambda m: "" if rng.random() < 0.6 else m.group(0), text)
    if "commas" in ops:
        text = text.replace(",", "")
    if "typo" in ops:
        words = text.split(" ")
        candidates = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
        for index in rng.sample(candidates, min(len(candidates), rng.choice((1, 1, 2)))):
            words[index] = _typo(words[index], rng)
        text = " ".join(words)
    return text


def _case(text: str, mode: str) -> str:
    if mode == "lower":
        return text.lower()
    if mode == "upper":
        return text.upper()
    return text


def augment(command: str, gold_text: str, rng: random.Random) -> tuple[str, str]:
    """Return (varied command, gold text as it now appears in it). ``gold_text`` may be empty."""
    start = command.find(gold_text) if gold_text else -1
    if start >= 0:
        pre, gold, post = command[:start], gold_text, command[start + len(gold_text) :]
    else:
        pre, gold, post = command, "", ""
    question = bool(QUESTION_START.match(command.strip()))
    first = command.strip().split(" ", 1)[0].casefold()
    if rng.random() < 0.4:
        imperative = first in VERBS and not question
        pool = PREFIXES_IMPERATIVE if imperative and rng.random() < 0.6 else PREFIXES_ANY
        pre = rng.choice(pool) + pre
    if rng.random() < 0.15 and not question:
        post = post + rng.choice(SUFFIXES)
    ops = {
        op
        for op, rate in (
            ("contract", 0.25), ("expand", 0.1), ("apostrophe", 0.12), ("textspeak", 0.12),
            ("british", 0.15), ("articles", 0.06), ("commas", 0.15), ("typo", 0.14),
        )  # fmt: skip
        if rng.random() < rate
    }
    pre, post = _token_noise(pre, rng, ops), _token_noise(post, rng, ops)
    if rng.random() < 0.05:
        pre = f"{rng.choice(FILLERS)} {pre}"
    end = rng.random()
    if question and end < 0.4:
        post += "?"
    elif not question and end < 0.08:
        post += "."
    elif not question and end < 0.11:
        post += "!"
    mode = rng.choices(("lower", "keep", "sentence", "upper"), (0.35, 0.38, 0.25, 0.02))[0]
    pre, gold, post = _case(pre, mode), _case(gold, mode), _case(post, mode)
    text = pre + gold + post
    if mode == "sentence":
        index = next((i for i, ch in enumerate(text) if ch.isalpha()), None)
        if index is not None:
            text = text[:index] + text[index].upper() + text[index + 1 :]
            if gold and len(pre) == index:
                gold = gold[0].upper() + gold[1:]
    return re.sub(r"\s+", " ", text).strip(), gold
