"""Run the Jevlet assistant: ``pythonw -m jevlet.app`` (tray icon + Alt+Space palette)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# pythonw has no console. Redirect before torch/transformers are imported: they create log
# handlers at import time, and a handler bound to a missing stderr hides every later error.
_LOG_FILE = Path(__file__).resolve().parents[2] / "data" / "jevlet.log"
if sys.stderr is None or sys.stdout is None:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(_LOG_FILE, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
# The model files are cached locally; never block startup on the network.
if (_LOG_FILE.parent / "model_cache").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import logging  # noqa: E402

from PySide6.QtCore import QAbstractNativeEventFilter, QRectF, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from jevlet.assistant.planner import Context

from .brain import DATA, Brain
from .palette import Palette, qcolor
from .theme import TOKENS as T
from .toast import Toast
from .win32 import MOD_ALT, MOD_CONTROL, MOD_SHIFT, VK_SPACE, is_hotkey_message, register_hotkey

HOTKEY_ID = 0x4A4C  # "JL"


class HotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, callback) -> None:
        super().__init__()
        self.callback = callback

    def nativeEventFilter(self, event_type, message):  # noqa: N802
        if event_type == b"windows_generic_MSG":
            hit, identifier = is_hotkey_message(message)
            if hit:
                logging.info("WM_HOTKEY id=%s", identifier)
            if hit and identifier == HOTKEY_ID:
                self.callback()
                return True, 0
        return False, 0


def tray_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(qcolor(T.accent), 7))
    painter.drawEllipse(QRectF(9, 9, 46, 46))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(qcolor(T.accent))
    painter.drawEllipse(QRectF(24, 24, 16, 16))
    painter.end()
    return QIcon(pixmap)


def _setup_logging() -> None:
    """pythonw has no console: send warnings, tracebacks, and Qt callback errors to a file."""
    DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=DATA / "jevlet.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for noisy in ("httpx", "huggingface_hub", "transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    sys.excepthook = lambda kind, value, trace: logging.error(
        "uncaught error", exc_info=(kind, value, trace)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DATA / "daily" / "current.pt"))
    args = parser.parse_args()
    _setup_logging()
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"No model at {args.checkpoint}; train and export one first.")

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("Jevlet")
    palette, toast = Palette(), Toast()
    own = lambda handle: handle in {int(palette.winId()), int(toast.winId())}  # noqa: E731
    state = {"teaching": False, "undo": None}

    brain = Brain(args.checkpoint)
    thread = QThread()
    brain.moveToThread(thread)
    thread.started.connect(brain.load)
    brain.ready.connect(palette.set_ready)
    brain.failed_to_load.connect(palette.set_load_error)
    palette.plan_requested.connect(brain.plan)
    palette.replan_requested.connect(brain.replan)
    brain.planned.connect(palette.show_plan)
    brain.failed.connect(palette.show_failure)

    def on_run(request, plan, context, command, chosen, original) -> None:
        palette.dismiss()
        QTimer.singleShot(60, lambda: brain_run(request, plan, context, command, chosen, original))

    def brain_run(*payload) -> None:
        QTimer.singleShot(0, brain, lambda: brain.run(*payload))

    palette.run_requested.connect(on_run)

    def on_executed(_request, outcome) -> None:
        if outcome.ok:
            state["undo"] = outcome.undo
            palette.undo_label = outcome.message if outcome.undo else ""
            toast.show_message(
                "success", outcome.message, "Alt+Space, Enter to undo" if outcome.undo else ""
            )
            plan = palette.plan
            if plan is not None and plan.skill.key == "timer" and plan.duration:
                label = outcome.message.replace("Timer set for ", "")
                QTimer.singleShot(
                    plan.duration * 1000,
                    lambda: tray.showMessage(
                        "Timer finished", f"{label} is up.", tray_icon(), 8000
                    ),
                )
        else:
            toast.show_message("error", outcome.message, hold_ms=3200)

    brain.executed.connect(on_executed)
    brain.failed.connect(
        lambda request, message: toast.show_message("error", "That didn't work", message, 4000)
    )

    def on_undo() -> None:
        undo = state["undo"]
        palette.dismiss()
        state["undo"], palette.undo_label = None, ""
        if undo is None:
            return
        try:
            undo()
            toast.show_message("success", "Undone")
        except Exception as error:  # noqa: BLE001
            toast.show_message("error", "Couldn't undo", str(error), 3200)

    palette.undo_requested.connect(on_undo)

    def on_teach(command, context) -> None:
        palette.dismiss()
        state["teaching"] = True
        toast.show_message(
            "teach",
            f"Show me the control for “{command}”",
            "Click it now  ·  Alt+Space cancels",
            hold_ms=0,
        )
        handle = context.current.handle if context and context.current else None
        QTimer.singleShot(0, brain, lambda: brain.teach(command, handle, own))

    def on_taught(demo) -> None:
        state["teaching"] = False
        if demo is None:
            toast.show_message("info", "Nothing recorded")
            return
        target = demo.target
        label = target["name"] or target["automation_id"] or target["control_type"]
        toast.show_message(
            "success",
            f"Learned: {target['control_type']} “{label}”",
            demo.process or demo.window_title,
            2200,
        )

    palette.teach_requested.connect(on_teach)
    brain.taught.connect(on_taught)

    def on_hotkey() -> None:
        logging.info("hotkey pressed")
        try:
            if state["teaching"]:
                brain.cancel_teaching()
                return
            exclude = (int(palette.winId()), int(toast.winId()))
            palette.summon(Context.capture(exclude_handles=exclude))
        except Exception:
            logging.exception("could not open the palette")
            toast.show_message("error", "Couldn't open Jevlet", "See data/jevlet.log", 4000)

    palette.winId()  # create the native window that receives WM_HOTKEY
    hotkey = register_hotkey(
        int(palette.winId()),
        HOTKEY_ID,
        [
            (MOD_ALT, VK_SPACE, "Alt+Space"),
            (MOD_ALT | MOD_SHIFT, VK_SPACE, "Alt+Shift+Space"),
            (MOD_CONTROL | MOD_ALT, ord("J"), "Ctrl+Alt+J"),
        ],
    )
    logging.info("hotkey registered: %s", hotkey or "none available")
    hotkey_filter = HotkeyFilter(on_hotkey)
    app.installNativeEventFilter(hotkey_filter)

    tray = QSystemTrayIcon(tray_icon())
    tray.setToolTip(f"Jevlet — {hotkey or 'no hotkey available'}")
    menu = QMenu()
    open_action = QAction(f"Open    {hotkey}", menu)
    open_action.triggered.connect(on_hotkey)
    quit_action = QAction("Quit Jevlet", menu)
    quit_action.triggered.connect(app.quit)
    menu.addAction(open_action)
    menu.addSeparator()
    menu.addAction(quit_action)
    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: on_hotkey() if reason == QSystemTrayIcon.ActivationReason.Trigger else None
    )
    tray.show()
    thread.start()
    toast.show_message(
        "info", f"Jevlet is running  ·  {hotkey or 'tray icon'} to open", hold_ms=2600
    )
    code = app.exec()
    thread.quit()
    thread.wait(2000)
    sys.exit(code)


if __name__ == "__main__":
    main()
