"""The model thread: planning, execution, and teaching never block the UI thread."""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from jevlet.assistant.planner import Context, Plan, Planner, execute, split_steps
from jevlet.feedback import FeedbackStore

DATA = Path(__file__).resolve().parents[2] / "data"


class Brain(QObject):
    ready = Signal(str)
    failed_to_load = Signal(str)
    planned = Signal(int, object, int)  # request id, Plan, total steps
    executed = Signal(int, object)  # request id, Outcome
    failed = Signal(int, str)
    taught = Signal(object)  # Demonstration or None

    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.planner: Planner | None = None
        self.feedback = FeedbackStore(DATA / "feedback.sqlite3")
        self._cancel_teach = threading.Event()

    @Slot()
    def load(self) -> None:
        try:
            import torch

            from jevlet.personalize import CALIBRATION_MINIMUM, validated_examples
            from jevlet.system_one import SystemOne

            device = "cuda" if torch.cuda.is_available() else "cpu"
            engine = SystemOne(
                self.checkpoint,
                device=device,
                calibrated=validated_examples(self.checkpoint) >= CALIBRATION_MINIMUM,
            )
            self.planner = Planner(engine)
            self.planner.plan("open notepad", Context(None, [], []))  # warm up kernels
            model = engine.model_id.split("/")[-1]
            self.ready.emit(f"{model}  ·  {'GPU' if device == 'cuda' else 'CPU'}")
        except Exception as error:  # noqa: BLE001 - surface any load failure in the UI
            self.failed_to_load.emit(f"{type(error).__name__}: {error}")

    @Slot(int, str, object)
    def plan(self, request: int, command: str, context: Context) -> None:
        if self.planner is None:
            return
        try:
            steps = split_steps(command) or [command]
            self.planned.emit(request, self.planner.plan(steps[0], context), len(steps))
        except Exception as error:  # noqa: BLE001
            self.failed.emit(request, str(error) or type(error).__name__)

    @Slot(int, object, object, object)
    def replan(self, request: int, skill, command: str, context: Context) -> None:
        """Fill arguments for an alternative the user picked with the arrow keys."""
        if self.planner is None:
            return
        try:
            plan = self.planner.plan_for(skill, command, context, 1.0, skill.risk_floor)
            self.planned.emit(request, plan, 1)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(request, str(error) or type(error).__name__)

    def _record(
        self, plan: Plan, context: Context, chosen_alternative: bool, original: Plan | None
    ) -> None:
        """An executed plan is a label: kept top choice = approval, alternative = correction."""
        from jevlet.assistant.skills import SKILL_QUESTION, SKILLS

        reference = original or plan
        options = tuple((s.name, s.description) for s in SKILLS)
        state = self.planner._state(plan.command, context) if self.planner else plan.command
        try:
            identifier = self.feedback.log_decision(
                state, SKILL_QUESTION, options, reference.skill.name
            )
            if chosen_alternative and reference.skill.name != plan.skill.name:
                self.feedback.submit_feedback(
                    identifier, approved=False, corrected_option=plan.skill.name
                )
            else:
                self.feedback.submit_feedback(identifier, approved=True)
        except ValueError:
            traceback.print_exc()

    @Slot(int, object, object, str, bool, object)
    def run(
        self,
        request: int,
        plan: Plan,
        context: Context,
        command: str,
        chosen_alternative: bool,
        original: object,
    ) -> None:
        try:
            outcome = execute(plan, context)
            self._record(plan, context, chosen_alternative, original)
            steps = split_steps(command)
            for step in steps[1:]:
                if not outcome.ok or self.planner is None:
                    break
                time.sleep(1.2)  # let the previous step's window appear
                fresh = Context.capture()
                next_plan = self.planner.plan(step, fresh)
                if next_plan.gate != "run":
                    self.planned.emit(request, next_plan, 1)  # hand the rest back to the user
                    return
                outcome = execute(next_plan, fresh)
            self.executed.emit(request, outcome)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(request, str(error) or type(error).__name__)

    @Slot(str, object, object)
    def teach(self, command: str, snapshot_window: int | None, ignore) -> None:
        from jevlet.desktop.demonstrations import DemonstrationStore, watch_for_demonstration

        self._cancel_teach.clear()
        try:
            demo = watch_for_demonstration(
                command, ignore, snapshot_window, timeout=30.0, cancel=self._cancel_teach
            )
            if demo is not None:
                DemonstrationStore(DATA / "demonstrations.sqlite3").record(demo)
            self.taught.emit(demo)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self.taught.emit(None)

    def cancel_teaching(self) -> None:
        self._cancel_teach.set()
