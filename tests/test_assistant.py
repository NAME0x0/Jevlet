from __future__ import annotations

import json
import random

import pytest

from jevlet.api import Choice, Noul
from jevlet.assistant import actions, planner
from jevlet.assistant.commands_synthetic import _command, _environment, generate_command_dataset
from jevlet.assistant.environment import App, Window
from jevlet.assistant.planner import Context, Plan, Planner, execute, split_steps
from jevlet.assistant.skills import NOT_APPLICABLE, SKILL_BY_KEY, SKILLS, Environment, slot_options
from jevlet.assistant.text import parse_duration, shortlist, span_candidates, strip_courtesy


def test_text_helpers() -> None:
    assert shortlist("launch my music app", ["Calculator", "Spotify", "Word"], 1) == ["Spotify"]
    assert "cheap flights to london" in span_candidates(
        "hey jevlet, search for cheap flights to london"
    )
    assert "summarize this article" in span_candidates("get Claude to summarize this article")
    assert "see you at 5" in span_candidates('type "see you at 5" in the chat')
    assert strip_courtesy("hey jevlet, can you open spotify") == "open spotify"
    assert parse_duration("remind me in an hour and a half") == 5400
    assert parse_duration("open spotify") is None


def test_shared_catalogue_matches_the_python_constants() -> None:
    # The C# app reads these from shared/skills.json; training uses the Python constants.
    from jevlet.assistant.skills import _DATA
    from jevlet.benchmarks import RISK_QUESTION, daily_state

    assert _DATA["risk_question"] == RISK_QUESTION
    assert _DATA["state_format"].format(command="x", window="y") == daily_state("x", "y")


def test_every_slot_offers_not_applicable_last() -> None:
    env = Environment(
        ["Spotify", "Word"], ["chrome: Docs", "Spotify: Spotify Premium"], "chrome: Docs"
    )
    for skill in SKILLS:
        for slot in skill.slots:
            options = slot_options(slot, "open spotify", env)
            assert options[-1] == NOT_APPLICABLE and len(options) == len(set(options))
    assert slot_options("window", "close it", env)[0] == "Current window (chrome: Docs)"
    assert "Bluetooth and devices" in slot_options("setting", "pair my headphones", env)


def _plan(skill: str, confidence=0.95, risk=0.02, args=None, missing="") -> Plan:
    plan = Plan("x", SKILL_BY_KEY[skill], confidence, risk, dict(args or {}))
    plan.arg_confidence = {slot: 0.9 for slot in plan.args}
    plan.missing = missing
    return plan


def test_gate_runs_only_confident_safe_plans() -> None:
    assert _plan("open_app", args={"app": "Spotify"}).gate == "run"
    assert _plan("open_app", confidence=0.6, args={"app": "Spotify"}).gate == "confirm"
    assert _plan("open_app", risk=0.2, args={"app": "Spotify"}).gate == "confirm"
    assert _plan("open_app", missing="app").gate == "clarify"
    assert _plan("clarify").gate == "clarify"


def test_split_steps() -> None:
    assert split_steps("open spotify then play the next song") == [
        "open spotify",
        "play the next song",
    ]
    assert split_steps("search for salt and pepper") == ["search for salt and pepper"]


class FakeEngine:
    """Answers the skill question with a fixed skill and every slot with its first option."""

    collator = None

    def __init__(self, skill: str) -> None:
        self.skill = SKILL_BY_KEY[skill].name

    def evaluate(self, state, questions):
        answers = {}
        for key, question in questions.items():
            if key == "risk":
                answers[key] = Noul(0.03, 0.97, None, False, 0.97)
                continue
            names = (
                tuple(question.criteria)
                if isinstance(question.criteria, dict)
                else tuple(question.criteria)
            )
            chosen = self.skill if key == "skill" else names[0]
            probabilities = tuple(
                0.9 if name == chosen else 0.1 / (len(names) - 1) for name in names
            )
            answers[key] = Choice(names, probabilities, chosen, 0.9)
        return answers


def test_planner_fills_arguments_from_live_options() -> None:
    context = Context(
        Window(11, "Inbox - Outlook", "OUTLOOK"),
        [Window(11, "Inbox - Outlook", "OUTLOOK"), Window(12, "Spotify Premium", "Spotify")],
        [App("Spotify", "Spotify.exe"), App("Word", "word.exe")],
    )
    plan = Planner(FakeEngine("open_app")).plan("open spotify", context)
    assert plan.skill.key == "open_app" and plan.args == {"app": "Spotify"}
    assert plan.gate == "run" and plan.title == "Open Spotify"
    assert [alt.key for alt, _ in plan.alternatives]
    minimize = Planner(FakeEngine("minimize_window")).plan("minimize this", context)
    assert minimize.title == "Minimize this window"


def test_execute_dispatches_to_the_right_native_action(monkeypatch) -> None:
    calls = []
    for name in ("launch_app", "window_state", "switch_to", "open_target", "web_search", "volume"):
        monkeypatch.setattr(
            actions,
            name,
            lambda *args, _n=name: calls.append((_n, args)) or actions.Outcome(True, _n),
        )
    context = Context(
        Window(11, "Inbox - Outlook", "OUTLOOK"),
        [Window(11, "Inbox - Outlook", "OUTLOOK"), Window(12, "Spotify Premium", "Spotify")],
        [App("Spotify", "SpotifyAB.Spotify")],
    )
    execute(_plan("open_app", args={"app": "Spotify"}), context)
    execute(_plan("close_window", args={"window": "Spotify: Spotify Premium"}), context)
    execute(
        _plan("minimize_window", args={"window": "Current window (OUTLOOK: Inbox - Outlook)"}),
        context,
    )
    execute(_plan("settings", args={"setting": "Bluetooth and devices"}), context)
    execute(_plan("search", args={"text": "rtx a2000 driver"}), context)
    assert calls == [
        ("launch_app", ("SpotifyAB.Spotify", "Spotify")),
        ("window_state", (12, "Spotify Premium", "close")),
        ("window_state", (11, "Inbox - Outlook", "minimize")),
        ("open_target", ("ms-settings:bluetooth", "Bluetooth and devices")),
        ("web_search", ("rtx a2000 driver",)),
    ]
    assert not execute(_plan("clarify"), context).ok
    assert planner.SAFE_TO_RUN == 0.9


def test_generated_commands_offer_their_gold_arguments(tmp_path) -> None:
    rng = random.Random(4)
    for skill in SKILLS:
        env, windows = _environment(rng, ())
        command, gold, risk = _command(rng, skill.key, env, windows)
        assert 0.0 <= risk <= 1.0 and set(gold) <= set(skill.slots + skill.optional_slots)
        for slot, value in gold.items():
            assert value in slot_options(slot, command, env), (skill.key, command, value)
    manifest = generate_command_dataset(tmp_path, counts=(300, 30), seed=5)
    stats = manifest["splits"]["train"]["stats"]
    assert sum(v for k, v in stats.items() if k.startswith("inserted")) <= 3
    assert stats["surface_varied"] > 250
    row = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    skill_question = row["questions"][0]
    assert skill_question["options"][skill_question["label"]].startswith(
        SKILL_BY_KEY[row["domain"]].name
    )


def test_assistant_benchmark_is_not_generatable(tmp_path) -> None:
    # Benchmark v1 was inspected while building v5 data and is contaminated by design; the
    # strict no-leak rule applies to v2, which was written before the v5/v6 phrasings.
    from jevlet.assistant import decontam
    from jevlet.assistant.benchmark_v2 import CASES

    assert all(decontam.closest_case(case.command) is not None for case in CASES)
    assert decontam.tokens("move standup to 10") == decontam.tokens("move standup to 11")
    generate_command_dataset(tmp_path, counts=(5000, 10), seed=11)
    for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines():
        assert decontam.closest_case(json.loads(line)["metadata"]["command"]) is None


def test_augment_keeps_the_gold_span_verbatim() -> None:
    from jevlet.assistant.augment import augment

    rng = random.Random(3)
    for _ in range(2000):
        command, gold = augment("remind me to call priya tomorrow at 6pm", "call priya", rng)
        assert gold.casefold() == "call priya" and gold in command


def test_lexicon_pools_avoid_benchmark_arguments() -> None:
    from jevlet.assistant import lexicon

    rng = random.Random(5)
    samples = [lexicon.task(rng) for _ in range(3000)] + [
        lexicon.event_title(rng) for _ in range(3000)
    ]
    samples += [lexicon.city(rng) for _ in range(500)] + [lexicon.place(rng) for _ in range(500)]
    assert not [s for s in samples if any(word in s.casefold() for word in lexicon.BLOCKED)]
    assert len(set(samples[:3000])) > 1000  # composed, not a short list


def test_topv2_parse_and_mapping() -> None:
    from jevlet.assistant.real_commands import locate, map_top, parse_top

    tree = (
        "[IN:CREATE_REMINDER Remind [SL:PERSON_REMINDED me ] to [SL:TODO call the bank ] "
        "[SL:DATE_TIME at 7 : 30 am ] ]"
    )
    intent, slots = parse_top(tree)
    assert intent == "CREATE_REMINDER" and ("TODO", ["call", "the", "bank"]) in slots
    assert locate("wake me at 7:30am", ["7", ":", "30", "am"]) == "7:30am"
    assert map_top("Remind me to call the bank at 7:30am", tree) == (
        "set_reminder",
        {"text": "call the bank"},
        0.03,
    )
    stopwatch = "[IN:CREATE_TIMER start the [SL:METHOD_TIMER stopwatch ] ]"
    assert map_top("start the stopwatch", stopwatch)[:2] == ("stopwatch", {"stopwatch": "Start"})
    assert map_top("text mom", "[IN:SEND_MESSAGE text [SL:RECIPIENT mom ] ]")[0] == "clarify"


def test_span_rules_cover_common_forms() -> None:
    cases = {
        "ping me this sunday at 11:20 to submit the expense claim": "submit the expense claim",
        "new task: renew my library card": "renew my library card",
        "type thank you": "thank you",
        "seoul time": "seoul",
        "claude please draft a complaint to the council": "draft a complaint to the council",
        "what's the weather in cairo?": "cairo",
    }
    for command, gold in cases.items():
        assert gold in span_candidates(command), (command, span_candidates(command))


def test_sharded_generation_is_worker_independent_and_loads_lazily(tmp_path) -> None:
    from jevlet.data import JsonlDecisionDataset

    one = generate_command_dataset(tmp_path / "one", counts=(6000, 10), seed=9, workers=1)
    two = generate_command_dataset(tmp_path / "two", counts=(6000, 10), seed=9, workers=2)
    assert one["splits"]["train"]["sha256"] == two["splits"]["train"]["sha256"]
    eager = JsonlDecisionDataset(tmp_path / "one" / "train.jsonl")
    lazy = JsonlDecisionDataset(tmp_path / "one" / "train.jsonl", lazy=True)
    assert len(lazy) == len(eager) == 6000
    assert lazy[5999] == eager[5999] and lazy[0] == eager[0]


@pytest.mark.parametrize("skill", [s.key for s in SKILLS])
def test_every_skill_has_a_generator(skill) -> None:
    command, _, _ = _command(random.Random(1), skill, *_environment(random.Random(1), ()))
    assert command.strip()
