"""The command palette: one input, one live interpretation, keyboard only.

Enter runs · Enter twice for anything risky · Down/Up for alternatives · Tab to show me ·
Esc to close · Enter on an empty bar undoes the last action.
"""

from __future__ import annotations

import time

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QGuiApplication,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from jevlet.assistant.planner import Context, Plan

from .theme import TOKENS, animations_enabled
from .win32 import bring_to_front, style_window

T = TOKENS
EXAMPLES = ("open spotify", "mute", "search for flights to london", "switch to outlook")


def qcolor(value: str) -> QColor:
    if value.startswith("rgba"):
        red, green, blue, alpha = (int(part) for part in value[5:-1].split(","))
        return QColor(red, green, blue, alpha)
    return QColor(value)


def font(family: str, size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    result = QFont(family)
    result.setPixelSize(size)
    result.setWeight(weight)
    result.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return result


class Mark(QWidget):
    """The Jevlet mark: an accent ring with a center dot (dim while the model loads)."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(20, 20)
        self.live = False

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = qcolor(T.accent) if self.live else qcolor(T.text_tertiary)
        painter.setPen(QPen(color, 2))
        painter.drawEllipse(QRectF(3, 3, 14, 14))
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(8, 8, 4, 4))


class ProgressLine(QWidget):
    """A 2 px line: a travelling accent segment while thinking, empty when idle."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(2)
        self.active = False
        self.phase = 0.0
        self.animation = QVariantAnimation(self, startValue=0.0, endValue=1.0, duration=900)
        self.animation.setLoopCount(-1)
        self.animation.valueChanged.connect(self._tick)

    def _tick(self, value) -> None:
        self.phase = float(value)
        self.update()

    def set_active(self, active: bool) -> None:
        self.active = active
        if active and animations_enabled():
            self.animation.start()
        else:
            self.animation.stop()
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self.active:
            return
        painter = QPainter(self)
        width = self.width()
        segment = width * 0.28
        left = -segment + (width + segment) * self.phase if self.animation.state() else 0
        painter.fillRect(
            QRectF(left, 0, segment if self.animation.state() else width, 2), qcolor(T.accent)
        )


class Row(QWidget):
    """An alternative: label left, likelihood right; keyboard-selected state is filled."""

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(T.space_3, 6, T.space_3, 6)
        self.label = QLabel()
        self.label.setFont(font(T.font, T.size_body))
        self.value = QLabel()
        self.value.setFont(font(T.font_mono, T.size_meta))
        layout.addWidget(self.label, 1)
        layout.addWidget(self.value)
        self.selected = False
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        text = T.text if selected else T.text_secondary
        self.label.setStyleSheet(f"color: {text};")
        self.value.setStyleSheet(f"color: {T.text_tertiary};")
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if self.selected:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(T.surface_selected))
            painter.drawRoundedRect(QRectF(self.rect()), T.radius_small, T.radius_small)


class Palette(QWidget):
    plan_requested = Signal(int, str, object)
    replan_requested = Signal(int, object, object, object)
    run_requested = Signal(int, object, object, str, bool, object)
    teach_requested = Signal(str, object)
    undo_requested = Signal()

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(T.width)
        self.request = 0
        self.plan: Plan | None = None
        self.original: Plan | None = None
        self.steps = 1
        self.context: Context | None = None
        self.armed = False
        self.selected_alternative = -1
        self.show_alternatives = False
        self.state = "loading"
        self.undo_label = ""
        self.ready = False
        self.requested_at = 0.0
        self._build()
        self._debounce = QTimer(self, singleShot=True, interval=90)
        self._debounce.timeout.connect(self._request_plan)
        self._render()

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(T.space_4, T.space_3 + 2, T.space_4, T.space_3)
        outer.setSpacing(T.space_2 + 2)

        top = QHBoxLayout()
        top.setSpacing(T.space_3)
        self.mark = Mark()
        self.input = QLineEdit()
        self.input.setFrame(False)
        self.input.setFont(font(T.font_display, T.size_input))
        self.input.setPlaceholderText("Tell Jevlet what to do")
        self.input.setStyleSheet(
            f"QLineEdit {{ background: transparent; color: {T.text}; border: none;"
            f" selection-background-color: {T.accent}; }}"
        )
        self.input.textEdited.connect(self._edited)
        self.input.installEventFilter(self)
        self.latency = QLabel()
        self.latency.setFont(font(T.font_mono, T.size_meta))
        self.latency.setStyleSheet(f"color: {T.text_tertiary};")
        top.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.input, 1)
        top.addWidget(self.latency, 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(top)

        self.progress = ProgressLine()
        outer.addWidget(self.progress)

        self.card = QWidget()
        card = QVBoxLayout(self.card)
        card.setContentsMargins(T.space_1, 0, T.space_1, 0)
        card.setSpacing(T.space_1)
        headline = QHBoxLayout()
        self.title = QLabel()
        self.title.setFont(font(T.font, T.size_title, QFont.Weight.DemiBold))
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.key = QLabel()
        self.key.setFont(font(T.font, T.size_meta, QFont.Weight.DemiBold))
        headline.addWidget(self.title, 1)
        headline.addWidget(self.key, 0, Qt.AlignmentFlag.AlignVCenter)
        self.meta = QLabel()
        self.meta.setFont(font(T.font, T.size_meta + 1))
        self.meta.setTextFormat(Qt.TextFormat.RichText)
        card.addLayout(headline)
        card.addWidget(self.meta)
        outer.addWidget(self.card)

        self.alternatives = QWidget()
        alternatives = QVBoxLayout(self.alternatives)
        alternatives.setContentsMargins(0, 0, 0, 0)
        alternatives.setSpacing(2)
        self.rows = [Row() for _ in range(3)]
        for row in self.rows:
            alternatives.addWidget(row)
        outer.addWidget(self.alternatives)

        self.hint = QLabel()
        self.hint.setFont(font(T.font, T.size_meta))
        self.hint.setStyleSheet(f"color: {T.text_tertiary};")
        outer.addWidget(self.hint)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(bounds, T.radius, T.radius)
        painter.fillPath(path, qcolor(T.surface))
        painter.setPen(QPen(qcolor(T.edge), 1))
        painter.drawPath(path)
        # A 1 px inner highlight along the top edge reads as a physical glass edge.
        painter.setPen(QPen(qcolor(T.edge_highlight), 1))
        painter.drawLine(int(T.radius), 1, int(self.width() - T.radius), 1)

    # ----------------------------------------------------------------- state
    def set_ready(self, label: str) -> None:
        self.ready = True
        self.mark.live = True
        self.mark.update()
        self.model_label = label
        if self.state == "loading":
            self.state = "empty"
        self._render()

    def set_load_error(self, message: str) -> None:
        self.state = "error"
        self.error = f"Model failed to load: {message}"
        self._render()

    def _edited(self, text: str) -> None:
        self.armed = False
        self.selected_alternative = -1
        self.show_alternatives = False
        self.original = None
        if not text.strip():
            self.state = "empty" if self.ready else "loading"
            self.plan = None
            self.progress.set_active(False)
            self._render()
            return
        if self.ready:
            self.state = "thinking"
            self.progress.set_active(True)
            self._debounce.start()
            self._render()

    def _request_plan(self) -> None:
        self.request += 1
        self.requested_at = time.perf_counter()
        self.plan_requested.emit(self.request, self.input.text().strip(), self.context)

    def show_plan(self, request: int, plan: Plan, steps: int) -> None:
        if request != self.request:
            return  # a newer keystroke superseded this plan
        self.progress.set_active(False)
        self.plan, self.steps = plan, steps
        self.state = {"run": "plan", "confirm": "plan", "clarify": "clarify"}[plan.gate]
        if plan.confidence < 0.6 and plan.gate != "clarify" and self.original is None:
            self.show_alternatives = True
        self._render()

    def show_failure(self, request: int, message: str) -> None:
        if request != self.request and request != -1:
            return
        self.progress.set_active(False)
        self.state = "error"
        self.error = message
        self._render()

    def _render(self) -> None:
        state, plan = self.state, self.plan
        self.alternatives.setVisible(False)
        self.latency.setText("")
        key_style = _key_style(T.text_tertiary)
        if state == "loading":
            self._set_card("Loading the model", "First launch takes a few seconds.", "", key_style)
        elif state == "empty":
            if self.undo_label:
                self._set_card(
                    f"Undo: {self.undo_label}", "Enter to undo the last action", "Enter", key_style
                )
            else:
                self._set_card("", "", "", key_style)
        elif state == "thinking":
            if plan is None:
                self._set_card(" ", " ", "", key_style)
        elif state in {"plan", "clarify"} and plan is not None:
            self._render_plan(plan, key_style)
        elif state == "error":
            self.title.setStyleSheet(f"color: {T.danger};")
            self.title.setText(self.error)
            self.meta.setText(
                f"<span style='color:{T.text_secondary}'>Try again, or rephrase the command.</span>"
            )
            self.key.setText("")
            self.key.setStyleSheet("")
        has_title = bool(self.title.text().strip())
        self.title.setVisible(has_title)
        self.key.setVisible(has_title and bool(self.key.text()))
        # An idle bar is just the input and a one-line hint, like the system search.
        self.card.setVisible(has_title or state == "thinking")
        self._render_alternatives()
        self.hint.setText(self._hint())
        self.adjustSize()

    def _set_card(self, title: str, meta: str, key: str, key_style: str) -> None:
        self.title.setStyleSheet(f"color: {T.text};")
        self.title.setText(title)
        self.meta.setText(f"<span style='color:{T.text_secondary}'>{meta}</span>")
        self.key.setText(key)
        self.key.setStyleSheet(key_style if key else "")

    def _render_plan(self, plan: Plan, key_style: str) -> None:
        confidence = round(100 * plan.confidence)
        latency = f"{plan.latency_ms:.0f} ms" if plan.latency_ms else ""
        self.latency.setText(latency)
        steps = (
            f" · then {self.steps - 1} more step{'s' if self.steps > 2 else ''}"
            if self.steps > 1
            else ""
        )
        if plan.gate == "clarify":
            missing = {"control": "which control", "duration": "how long", "text": "what text"}.get(
                plan.missing, ""
            )
            detail = f"I can't tell {missing}." if missing else "I'm not sure what you mean."
            self.title.setStyleSheet(f"color: {T.text};")
            self.title.setText(detail)
            advice = "Rephrase it, or press Tab and show me."
            self.meta.setText(f"<span style='color:{T.text_secondary}'>{advice}</span>")
            self.key.setText("Tab")
            self.key.setStyleSheet(key_style)
            return
        risky = plan.risk >= 0.5
        if plan.gate == "run":
            status = f"<span style='color:{T.text_tertiary}'>Safe to run</span>"
            key, key_color = "Enter", T.accent
        elif self.armed:
            status = f"<span style='color:{T.warning}'>Press Enter again to do it</span>"
            key, key_color = "Enter again", T.warning
        else:
            reason = "Risky" if risky else "Not fully sure"
            color = T.danger if risky else T.warning
            status = f"<span style='color:{color}'>{reason} · needs a second Enter</span>"
            key, key_color = "Enter ×2", color
        self.title.setStyleSheet(f"color: {T.warning if self.armed else T.text};")
        self.title.setText(plan.title + steps)
        self.meta.setText(
            f"<span style='color:{T.text_secondary}'>{plan.skill.name}</span>"
            f"<span style='color:{T.text_tertiary}'>  ·  {confidence}% sure  ·  </span>{status}"
        )
        self.key.setText(key)
        self.key.setStyleSheet(_key_style(key_color))

    def _render_alternatives(self) -> None:
        plan = self.plan
        visible = bool(
            plan
            and self.show_alternatives
            and plan.alternatives
            and self.state in {"plan", "clarify"}
        )
        self.alternatives.setVisible(visible)
        if not visible:
            return
        for index, row in enumerate(self.rows):
            if index < len(plan.alternatives):
                skill, probability = plan.alternatives[index]
                row.label.setText(skill.name)
                row.value.setText(f"{100 * probability:.0f}%")
                row.set_selected(index == self.selected_alternative)
                row.show()
            else:
                row.hide()

    def _hint(self) -> str:
        if self.state == "loading":
            return ""
        if self.state == "empty":
            if self.undo_label:
                return f"Esc close   ·   {getattr(self, 'model_label', '')}"
            return "Try   " + "   ·   ".join(f"“{example}”" for example in EXAMPLES)
        if self.state in {"error", "thinking"}:
            return "Esc close"
        parts = ["Tab show me", "Esc close"]
        if self.plan and self.plan.alternatives:
            parts.insert(0, "↓ alternatives")
        return "   ·   ".join(parts)

    # ------------------------------------------------------------- keyboard
    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self.input and event.type() == QEvent.Type.KeyPress:
            return self._key(event)
        return super().eventFilter(watched, event)

    def _key(self, event: QKeyEvent) -> bool:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self.armed:
                self.armed = False
                self._render()
            else:
                self.dismiss()
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._enter()
            return True
        if key == Qt.Key.Key_Tab:
            text = self.input.text().strip()
            if text:
                self.teach_requested.emit(text, self.context)
            return True
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.plan and self.plan.alternatives:
            self.show_alternatives = True
            count = len(self.plan.alternatives)
            step = 1 if key == Qt.Key.Key_Down else -1
            self.selected_alternative = max(-1, min(count - 1, self.selected_alternative + step))
            self.armed = False
            self._render()
            return True
        return False

    def _enter(self) -> None:
        if self.state == "empty" and self.undo_label:
            self.undo_requested.emit()
            return
        plan = self.plan
        if plan is None or self.state not in {"plan", "clarify"}:
            return
        if self.selected_alternative >= 0:
            skill = plan.alternatives[self.selected_alternative][0]
            self.original = self.original or plan
            self.selected_alternative = -1
            self.show_alternatives = False
            self.state = "thinking"
            self.progress.set_active(True)
            self.request += 1
            self.replan_requested.emit(self.request, skill, self.input.text().strip(), self.context)
            self._render()
            return
        if plan.gate == "clarify":
            return
        if plan.gate == "confirm" and not self.armed:
            self.armed = True
            self._render()
            return
        self.armed = False
        self.run_requested.emit(
            self.request,
            plan,
            self.context,
            self.input.text().strip(),
            self.original is not None,
            self.original,
        )

    # ------------------------------------------------------------ visibility
    def summon(self, context: Context) -> None:
        self.context = context
        if self.isVisible():
            self.dismiss()
            return
        self.input.selectAll()
        if self.state not in {"loading"}:
            self.state = "empty" if not self.input.text().strip() else self.state
        self._render()
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        target = QPoint(area.center().x() - T.width // 2, area.top() + int(area.height() * 0.22))
        self.setWindowOpacity(0.0 if animations_enabled() else 1.0)
        self.move(target + QPoint(0, 8))
        self.show()
        style_window(int(self.winId()))
        bring_to_front(int(self.winId()))
        self.activateWindow()
        self.input.setFocus()
        if self.input.text().strip() and self.ready:
            self._edited(self.input.text())
        if animations_enabled():
            self._animate(target)
        else:
            self.move(target)

    def _animate(self, target: QPoint) -> None:
        fade = QPropertyAnimation(self, b"windowOpacity", self)
        fade.setDuration(T.enter_ms)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        slide = QPropertyAnimation(self, b"pos", self)
        slide.setDuration(T.enter_ms)
        slide.setStartValue(target + QPoint(0, 8))
        slide.setEndValue(target)
        slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        fade.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        slide.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def dismiss(self) -> None:
        self.armed = False
        self.progress.set_active(False)
        if not self.isVisible():
            return
        if animations_enabled():
            fade = QPropertyAnimation(self, b"windowOpacity", self)
            fade.setDuration(T.exit_ms)
            fade.setStartValue(self.windowOpacity())
            fade.setEndValue(0.0)
            fade.finished.connect(self.hide)
            fade.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        else:
            self.hide()

    def changeEvent(self, event) -> None:  # noqa: N802
        if (
            event.type() == QEvent.Type.ActivationChange
            and not self.isActiveWindow()
            and self.isVisible()
        ):
            self.dismiss()
        super().changeEvent(event)


def _key_style(color: str) -> str:
    """A keycap: the key hint uses the state color on a raised surface."""
    return f"color: {color}; padding: 2px 8px; border-radius: 5px; background: {T.surface_raised};"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
