from __future__ import annotations

import os

import pytest

from jevlet.desktop.demonstrations import (
    ClickRecorder,
    Demonstration,
    DemonstrationStore,
    demonstration_example,
)
from jevlet.grounding import NONE_OF_THESE


def _control(name: str, automation_id: str, kind: str) -> dict:
    return {
        "name": name,
        "automation_id": automation_id,
        "control_type": kind,
        "rect": [0, 0, 10, 10],
        "enabled": True,
    }


CONTROLS = (
    _control("Reply", "reply", "Button"),
    _control("Forward", "", "Button"),
    _control("", "", "Pane"),
)


def test_store_round_trip_and_example_labels_the_clicked_control(tmp_path) -> None:
    store = DemonstrationStore(tmp_path / "demos.sqlite3")
    demo = Demonstration("answer Omar", "Inbox - Outlook", "OUTLOOK", CONTROLS, CONTROLS[0])
    identifier = store.record(demo)
    [(stored_id, stored)] = store.all()
    assert stored_id == identifier and stored == demo
    example = demonstration_example(identifier, stored)
    question = example.questions[0]
    assert question.options == ["Button: Reply", "Button: Forward", NONE_OF_THESE]
    assert question.options[question.label] == "Button: Reply"
    assert "answer Omar" in example.state and "Outlook" in example.state


def test_clicked_control_missing_from_snapshot_is_added_before_none_option() -> None:
    target = {
        "name": "Archive",
        "automation_id": "",
        "control_type": "Button",
        "rect": [1, 1, 2, 2],
        "enabled": True,
    }
    example = demonstration_example(
        1, Demonstration("archive it", "Inbox", "OUTLOOK", CONTROLS, target)
    )
    options = example.questions[0].options
    assert options[-1] == NONE_OF_THESE and options[example.questions[0].label] == "Button: Archive"


def test_unlabeled_surface_click_is_not_a_training_example() -> None:
    target = {
        "name": "",
        "automation_id": "",
        "control_type": "Document",
        "rect": [1, 1, 2, 2],
        "enabled": True,
    }
    assert demonstration_example(1, Demonstration("x", "w", "p", CONTROLS, target)) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows hook")
def test_hook_installs_and_uninstalls_without_clicks() -> None:
    recorder = ClickRecorder()
    recorder.start()
    assert recorder.next_click(lambda handle: False, timeout=0.2) is None
    recorder.stop()
    assert recorder._thread is None
