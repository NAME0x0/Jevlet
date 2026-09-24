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
        assert 0.0 <= risk <= 1.0 and set(gold) <= set(skill.slots)
        for slot, value in gold.items():
            assert value in slot_options(slot, command, env), (skill.key, command, value)
    manifest = generate_command_dataset(tmp_path, counts=(300, 30), seed=5)
    assert (
        sum(v for k, v in manifest["splits"]["train"]["stats"].items() if k.startswith("inserted"))
        <= 3
    )
    row = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    skill_question = row["questions"][0]
    assert skill_question["options"][skill_question["label"]].startswith(
        SKILL_BY_KEY[row["domain"]].name
    )


def test_assistant_benchmark_is_not_generatable() -> None:
    import re

    from jevlet.assistant.benchmark import CASES

    def tokens(text: str) -> frozenset[str]:
        return frozenset(re.findall(r"[a-z0-9']+", text.casefold()))

    rng = random.Random(0)
    generated = set()
    for _ in range(20000):
        skill = rng.choice(SKILLS).key
        generated.add(tokens(_command(rng, skill, *_environment(rng, ()))[0]))
    for case in CASES:
        mine = tokens(case.command)
        closest = max(len(mine & other) / len(mine | other) for other in generated)
        assert closest < 0.8, case.command


@pytest.mark.parametrize("skill", [s.key for s in SKILLS])
def test_every_skill_has_a_generator(skill) -> None:
    command, _, _ = _command(random.Random(1), skill, *_environment(random.Random(1), ()))
    assert command.strip()
