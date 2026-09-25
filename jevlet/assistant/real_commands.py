"""Human-written assistant commands (TOPv2, MASSIVE, CLINC150) mapped onto Jevlet's skills.

Synthetic templates carry the author's phrasing habits; these corpora carry thousands of other
people's. Only intents with an unambiguous Jevlet skill are mapped; the rest are skipped rather
than forced. TOPv2's semantic parses give verbatim spans for text arguments (reminder task,
destination, music, alarm name); a span the extractor does not offer is not inserted: the row
keeps its skill question only and the miss is counted, which measures real-world span recall.

Splits follow the sources: train -> train, eval/validation -> dev, test -> vault (never trained
on). Rows within the benchmark_v2 decontamination threshold are dropped.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from jevlet.benchmarks import RISK_QUESTION, daily_state
from jevlet.data import DecisionExample, Question

from . import decontam
from .augment import augment
from .commands_synthetic import _environment, _golds_offered
from .skills import NOT_APPLICABLE, SKILL_BY_KEY, SKILL_QUESTION, SKILLS, SLOTS, slot_options

RISK = {
    "set_alarm": 0.02, "list_alarms": 0.01, "cancel_alarm": 0.12, "alarm_control": 0.01, "set_reminder": 0.03,
    "list_reminders": 0.01, "cancel_reminder": 0.1, "timer": 0.01, "timer_control": 0.02, "stopwatch": 0.01,
    "play_music": 0.02, "media": 0.01, "weather": 0.01, "directions": 0.02, "search": 0.01, "volume": 0.01,
    "create_event": 0.05, "show_agenda": 0.01, "cancel_event": 0.45, "today": 0.01, "world_time": 0.01,
    "compose_email": 0.1, "add_todo": 0.02, "show_todos": 0.01, "complete_todo": 0.03, "calculate": 0.01,
}  # fmt: skip
MUSIC_SLOTS = {
    "MUSIC_TYPE", "MUSIC_ARTIST_NAME", "MUSIC_GENRE", "MUSIC_TRACK_TITLE", "MUSIC_PLAYLIST_TITLE",
    "MUSIC_ALBUM_TITLE", "MUSIC_RADIO_ID", "MUSIC_ALBUM_MODIFIER", "MUSIC_PLAYLIST_MODIFIER",
}  # fmt: skip
TOP_FIXED = {
    "GET_ALARM": ("list_alarms", {}), "DELETE_ALARM": ("cancel_alarm", None),
    "SILENCE_ALARM": ("alarm_control", {"alarm_action": "Stop ringing"}),
    "SNOOZE_ALARM": ("alarm_control", {"alarm_action": "Snooze"}),
    "GET_REMINDER": ("list_reminders", {}), "DELETE_REMINDER": ("cancel_reminder", None),
    "PAUSE_TIMER": ("timer_control", {"timer_action": "Pause"}), "RESUME_TIMER": ("timer_control", {"timer_action": "Resume"}),
    "DELETE_TIMER": ("timer_control", {"timer_action": "Cancel"}), "GET_TIMER": ("timer_control", {"timer_action": "Time left"}),
    "ADD_TIME_TIMER": ("timer_control", {"timer_action": "Add time"}),
    "PAUSE_MUSIC": ("media", {"media": "Play or pause"}), "STOP_MUSIC": ("media", {"media": "Play or pause"}),
    "SKIP_TRACK_MUSIC": ("media", {"media": "Next track"}), "PREVIOUS_TRACK_MUSIC": ("media", {"media": "Previous track"}),
}  # fmt: skip
STOPWATCH = {"CREATE_TIMER": "Start", "PAUSE_TIMER": "Stop", "DELETE_TIMER": "Reset", "RESUME_TIMER": "Start"}
NAVIGATION = {
    "GET_DIRECTIONS", "GET_ESTIMATED_DURATION", "GET_DISTANCE", "GET_ESTIMATED_ARRIVAL", "GET_INFO_TRAFFIC",
    "GET_ESTIMATED_DEPARTURE", "GET_INFO_ROUTE",
}  # fmt: skip
MASSIVE = {
    "alarm_set": ("set_alarm", None), "alarm_query": ("list_alarms", {}), "alarm_remove": ("cancel_alarm", None),
    "audio_volume_up": ("volume", {"volume": "Volume up"}), "audio_volume_down": ("volume", {"volume": "Volume down"}),
    "audio_volume_mute": ("volume", {"volume": "Mute or unmute"}), "datetime_convert": ("world_time", None),
    "email_sendemail": ("compose_email", None), "lists_createoradd": ("add_todo", None), "lists_query": ("show_todos", {}),
    "lists_remove": ("complete_todo", None), "play_music": ("play_music", None), "qa_maths": ("calculate", None),
    "qa_currency": ("calculate", None), "qa_factoid": ("search", None), "qa_definition": ("search", None),
    "news_query": ("search", None), "cooking_recipe": ("search", None), "recommendation_events": ("search", None),
    "recommendation_locations": ("search", None), "recommendation_movies": ("search", None),
    "weather_query": ("weather", None), "transport_traffic": ("directions", None),
}  # fmt: skip
MASSIVE_CLARIFY = {
    "takeaway_order": 0.9, "transport_ticket": 0.9, "transport_taxi": 0.85, "social_post": 0.9,
    "iot_hue_lightdim": 0.1, "iot_hue_lightup": 0.1, "iot_hue_lightoff": 0.1, "iot_hue_lighton": 0.1,
    "iot_hue_lightchange": 0.1, "iot_wemo_on": 0.15, "iot_wemo_off": 0.15, "iot_cleaning": 0.1, "iot_coffee": 0.1,
    "general_quirky": 0.05, "general_joke": 0.05, "general_greet": 0.05,
}  # fmt: skip
CLINC = {
    "alarm": ("set_alarm", None), "timer": ("timer", {}), "reminder": ("set_reminder", None),
    "calendar": ("show_agenda", {}), "todo_list": ("show_todos", {}), "calculator": ("calculate", None),
    "measurement_conversion": ("calculate", None), "time": ("today", {}), "date": ("today", {}),
    "timezone": ("world_time", None), "weather": ("weather", None), "directions": ("directions", None),
    "traffic": ("directions", None), "distance": ("directions", None), "play_music": ("play_music", None),
    "next_song": ("media", {"media": "Next track"}), "definition": ("search", None),
}  # fmt: skip
CLINC_CLARIFY = {
    "text": 0.75, "make_call": 0.5, "book_flight": 0.9, "book_hotel": 0.9, "uber": 0.85, "car_rental": 0.9,
    "order": 0.9, "pay_bill": 0.95, "transfer": 0.97, "restaurant_reservation": 0.8, "share_location": 0.85,
    "freeze_account": 0.95, "report_lost_card": 0.9, "pin_change": 0.95, "cancel_reservation": 0.8,
}  # fmt: skip


def parse_top(tree: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """(root intent, [(top-level slot, its words)]) from a TOPv2 bracketed parse."""
    tokens = tree.split()
    intent = tokens[0].removeprefix("[IN:")
    slots: list[tuple[str, list[str]]] = []
    depth, current = 0, None
    for token in tokens[1:]:
        if token.startswith("["):
            depth += 1
            if depth == 1 and token.startswith("[SL:"):
                current = (token.removeprefix("[SL:"), [])
            continue
        if token == "]":
            if depth == 1 and current is not None:
                slots.append(current)
                current = None
            depth -= 1
            continue
        if current is not None:
            current[1].append(token)
    return intent, slots


def locate(utterance: str, words: list[str]) -> str | None:
    """The verbatim utterance span for parse words ("7 : 30 am" -> "7:30am")."""
    if not words:
        return None
    match = re.search(r"\s*".join(re.escape(word) for word in words), utterance, re.I)
    return utterance[match.start() : match.end()] if match else None


def _music_span(utterance: str, slots: list[tuple[str, list[str]]]) -> str | None:
    spans = [
        match
        for name, words in slots
        if name in MUSIC_SLOTS
        for match in [re.search(r"\s*".join(re.escape(w) for w in words), utterance, re.I)]
        if match
    ]
    if not spans:
        return None
    return utterance[min(m.start() for m in spans) : max(m.end() for m in spans)]


def map_top(utterance: str, tree: str) -> tuple[str, dict[str, str] | None, float] | None:
    """(skill, gold args or None for skill-only, P(risky)) or None when unmapped."""
    intent, slots = parse_top(tree)
    by_name = {name: words for name, words in slots}
    method = " ".join(by_name.get("METHOD_TIMER", [])).casefold()
    if intent in STOPWATCH and "stopwatch" in method:
        return "stopwatch", {"stopwatch": STOPWATCH[intent]}, 0.01
    if intent == "CREATE_TIMER":
        return "timer", {}, 0.01
    if intent in TOP_FIXED:
        skill, gold = TOP_FIXED[intent]
        return skill, gold, RISK[skill]
    if intent == "CREATE_ALARM":
        name = locate(utterance, by_name.get("ALARM_NAME", []))
        return "set_alarm", ({"text": name} if name else {}), RISK["set_alarm"]
    if intent == "CREATE_REMINDER":
        person = " ".join(by_name.get("PERSON_REMINDED", ["me"])).casefold()
        if person not in {"me", "myself", "i"}:
            return None
        task = locate(utterance, by_name.get("TODO", []))
        return "set_reminder", ({"text": task} if task else None), RISK["set_reminder"]
    if intent == "PLAY_MUSIC":
        music = _music_span(utterance, slots)
        return "play_music", ({"text": music} if music else None), RISK["play_music"]
    if intent in {"GET_WEATHER", "GET_SUNSET", "GET_SUNRISE"}:
        place = locate(utterance, by_name.get("LOCATION", []))
        return "weather", ({"text": place} if place else {}), RISK["weather"]
    if intent in NAVIGATION:
        target = locate(utterance, by_name.get("DESTINATION", []) or by_name.get("LOCATION", []))
        return "directions", ({"text": target} if target else None), RISK["directions"]
    if intent == "GET_EVENT" and ("CATEGORY_EVENT" in by_name or "NAME_EVENT" in by_name):
        return "search", None, RISK["search"]
    if intent == "SEND_MESSAGE":
        return "clarify", {}, 0.75
    if intent in {"GET_MESSAGE", "REACT_MESSAGE"}:
        return "clarify", {}, 0.2
    return None


def map_massive(label: str, text: str) -> tuple[str, dict[str, str] | None, float] | None:
    remind = "remind" in text.casefold()
    if label == "calendar_set":
        return ("set_reminder" if remind else "create_event"), None, 0.04
    if label == "calendar_query":
        return ("list_reminders" if remind else "show_agenda"), {}, 0.01
    if label == "calendar_remove":
        return ("cancel_reminder" if remind else "cancel_event"), None, 0.3
    if label == "datetime_query":
        world = re.search(r"\btime (?:in|at)\b|\btime ?zone\b", text, re.I)
        return ("world_time", None, 0.01) if world else ("today", {}, 0.01)
    if label in MASSIVE:
        skill, gold = MASSIVE[label]
        return skill, gold, RISK[skill]
    if label in MASSIVE_CLARIFY:
        return "clarify", {}, MASSIVE_CLARIFY[label]
    return None


def map_clinc(label: str, text: str) -> tuple[str, dict[str, str] | None, float] | None:
    if label in CLINC:
        skill, gold = CLINC[label]
        return skill, gold, RISK[skill]
    if label in CLINC_CLARIFY:
        return "clarify", {}, CLINC_CLARIFY[label]
    return None


def _example(
    rng: random.Random, row_id: str, command: str, skill_key: str, gold: dict[str, str] | None,
    risk: float, split: str, source: str, stats: Counter, vary: bool,
) -> DecisionExample | None:  # fmt: skip
    env, _ = _environment(rng, ())
    skill = SKILL_BY_KEY[skill_key]
    if gold is not None and "text" in gold and gold["text"] not in slot_options("text", command, env):
        stats[f"span_miss:{skill_key}"] += 1
        gold = None  # keep the skill question; never teach an option the app would not offer
    if vary and gold is not None:
        varied, text = augment(command, gold.get("text", ""), rng)
        varied_gold = {**gold, "text": text} if "text" in gold else dict(gold)
        if (not gold.get("text") or text) and _golds_offered(varied, varied_gold, env):
            command, gold = varied, varied_gold
    elif vary:
        command, _ = augment(command, "", rng)
    if decontam.closest_case(command) is not None:
        stats["decontaminated"] += 1
        return None
    skills = list(SKILLS)
    if rng.random() < 0.5:
        rng.shuffle(skills)
    names = [s.name for s in skills]
    questions = [
        Question(SKILL_QUESTION, names, names.index(skill.name)),
        Question(RISK_QUESTION, ["True", "False"], 0 if risk >= 0.5 else 1, "noul", [risk, 1 - risk]),
    ]
    if gold is not None:
        for slot in skill.slots + skill.optional_slots:
            answer = gold.get(slot, NOT_APPLICABLE) if slot in skill.optional_slots else gold.get(slot)
            if answer is None:
                continue  # a required argument the source does not pin down (e.g. which alarm)
            options = slot_options(slot, command, env)
            if answer not in options:
                stats[f"slot_miss:{slot}"] += 1
                continue
            questions.append(Question(SLOTS[slot].question, options, options.index(answer)))
    stats[f"{source}:{skill_key}"] += 1
    return DecisionExample(
        row_id, daily_state(command, env.current_window), questions, "assistant", skill_key, split,
        metadata={"command": command, "source": source, "gold": gold or {}},
    )  # fmt: skip


def _top_rows(root: Path, split: str) -> list[tuple[str, str, str]]:
    import pandas as pd

    frame = pd.read_parquet(root / "default" / split / "0000.parquet")
    return list(zip(frame["utterance"], frame["semantic_parse"], frame["domain"], strict=True))


def _hf_rows(repo: str, config: str | None, split: str, text: str, label: str) -> list[tuple[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset(repo, config, split=split)
    feature = dataset.features.get(label)
    names = getattr(feature, "names", None)
    return [(row[text], names[row[label]] if names else row[label]) for row in dataset]


def generate_real_dataset(
    output_dir: str | Path, top_root: str | Path = "data/raw/top_v2", seed: int = 31,
    variants: int = 2, caps: dict[str, int] | None = None,
) -> dict:  # fmt: skip
    """Write train/dev/vault JSONL plus ``vault/benchmark.jsonl`` (command, skill, gold)."""
    caps = caps or {"weather": 12_000, "directions": 12_000, "clarify": 6_000, "search": 4_000}
    root = Path(output_dir)
    (root / "vault").mkdir(parents=True, exist_ok=True)
    plan = {
        "train": ("train", "train", "train", variants),
        "dev": ("eval", "validation", "validation", 1),
        "vault": ("test", "test", "test", 1),
    }
    manifest: dict = {"seed": seed, "variants": variants, "caps": caps, "splits": {}}
    for split, (top_split, massive_split, clinc_split, copies) in plan.items():
        rng = random.Random(f"{seed}:{split}")
        stats: Counter = Counter()
        sources: list[tuple[str, str, tuple]] = []
        for utterance, tree, _ in _top_rows(Path(top_root), top_split):
            sources.append(("topv2", utterance, map_top(utterance, tree) or ()))
        for text, label in _hf_rows("mteb/amazon_massive_intent", "en", massive_split, "text", "label_text"):
            sources.append(("massive", text, map_massive(label, text) or ()))
        for text, label in _hf_rows("clinc/clinc_oos", "plus", clinc_split, "text", "intent"):
            sources.append(("clinc", text, map_clinc(label, text) or ()))
        mapped = [(source, text, target) for source, text, target in sources if target]
        stats["unmapped"] = len(sources) - len(mapped)
        rng.shuffle(mapped)
        per_skill: Counter = Counter()
        rows, benchmark = [], []
        for index, (source, text, (skill, gold, risk)) in enumerate(mapped):
            cap = caps.get(skill)
            if cap is not None and per_skill[skill] >= (cap if split == "train" else cap // 8):
                stats[f"capped:{skill}"] += 1
                continue
            per_skill[skill] += 1
            for copy in range(copies):
                example = _example(
                    rng, f"{split}-real-{index}-{copy}", text.strip(), skill, gold, risk, split, source,
                    stats, vary=copy > 0,
                )  # fmt: skip
                if example is not None:
                    rows.append(json.dumps(example.to_dict(), sort_keys=True) + "\n")
                    if split == "vault" and copy == 0:
                        benchmark.append({"command": text.strip(), "skill": skill, "args": example.metadata["gold"], "source": source})
        path = root / ("vault" if split == "vault" else "") / f"{split}.jsonl"
        path.write_text("".join(rows), encoding="utf-8", newline="\n")
        manifest["splits"][split] = {
            "rows": len(rows),
            "skills": dict(sorted(per_skill.items())),
            "stats": dict(sorted(stats.items())),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if benchmark:
            (root / "vault" / "benchmark.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in benchmark), encoding="utf-8", newline="\n"
            )  # fmt: skip
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
