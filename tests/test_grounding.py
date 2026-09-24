from __future__ import annotations

import json
import random

from jevlet.desktop.native import ControlInfo
from jevlet.grounding import NONE_OF_THESE, actionable, control_text, grounding_options
from jevlet.grounding_synthetic import APPS, HELD_OUT, _example, generate_grounding_dataset


def test_every_task_targets_a_listed_control() -> None:
    for app in APPS:
        names = [name for _, name in app.controls]
        assert len(names) == len(set(names)), app.name
        assert set(app.tasks) <= set(names), (app.name, set(app.tasks) - set(names))
        assert app.risky <= set(app.tasks)


def test_labels_point_at_target_and_none_option_is_last() -> None:
    rng = random.Random(3)
    for index in range(500):
        example = _example(rng, APPS, "train", index)
        question = example.questions[0]
        assert question.options[-1] == NONE_OF_THESE
        assert len(question.options) == len(set(question.options))
        selected = question.options[question.label]
        if example.metadata["foreign"]:
            assert selected == NONE_OF_THESE and question.is_unknown
        else:
            assert selected.endswith(": " + example.metadata["target"])


def test_held_out_apps_never_reach_training(tmp_path) -> None:
    manifest = generate_grounding_dataset(tmp_path, counts=(400, 100), seed=1)
    assert not set(manifest["splits"]["train"]["apps"]) & HELD_OUT
    assert set(manifest["splits"]["dev"]["apps"]) == HELD_OUT
    rows = [json.loads(line) for line in (tmp_path / "dev.jsonl").read_text().splitlines()]
    assert all(row["is_ood"] for row in rows)


def test_captured_inventory_becomes_trainable_app(tmp_path) -> None:
    from jevlet.grounding_live import inventory_apps

    (tmp_path / "notes.json").write_text(
        json.dumps(
            {
                "process": "NotesApp",
                "titles": ["{weird} title"],
                "controls": [
                    ["Button", "Save"],
                    ["Button", "Delete"],
                    ["Button", "Settings"],
                    ["Edit", "Body"],
                    ["Button", "Copy {x}"],
                ],
            }
        )
    )
    [app] = inventory_apps(tmp_path)
    assert set(app.tasks) == {"Save", "Delete", "Settings"}
    assert app.risky == {"Delete"}
    rng = random.Random(0)
    for index in range(50):
        example = _example(rng, (app,), "train", index)
        assert "{" not in example.state


def test_live_controls_are_filtered_and_formatted_like_training_data() -> None:
    controls = [
        ControlInfo("", "", "Pane"),
        ControlInfo("Reply", "reply", "Button"),
        ControlInfo("Reply", "reply2", "Button"),
        ControlInfo("Search", "", "Edit"),
        ControlInfo("Disabled", "", "Button", enabled=False),
        ControlInfo("", "CloseButton", "Button"),
        ControlInfo("Inbox - Outlook", "", "Text"),
    ]
    kept = actionable(controls)
    assert grounding_options(kept) == [
        "Button: Reply",
        "Edit: Search",
        "Button: [CloseButton]",
        NONE_OF_THESE,
    ]
    assert control_text("Button", "Reply") == "Button: Reply"
