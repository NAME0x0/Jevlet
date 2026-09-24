"""A small status pill that never takes focus: results, errors, and teaching prompts."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, QTimer
from PySide6.QtGui import QCursor, QFont, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .palette import font, qcolor
from .theme import TOKENS as T

GLYPHS = {"success": T.success, "error": T.danger, "teach": T.accent, "info": T.text_tertiary}


class Toast(QWidget):
    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.kind = "info"
        layout = QHBoxLayout(self)
        layout.setContentsMargins(T.space_4 + 14, T.space_2 + 2, T.space_4, T.space_2 + 2)
        text = QVBoxLayout()
        text.setSpacing(0)
        self.message = QLabel()
        self.message.setFont(font(T.font, T.size_body, QFont.Weight.DemiBold))
        self.message.setStyleSheet(f"color: {T.text};")
        self.detail = QLabel()
        self.detail.setFont(font(T.font, T.size_meta))
        self.detail.setStyleSheet(f"color: {T.text_tertiary};")
        text.addWidget(self.message)
        text.addWidget(self.detail)
        layout.addLayout(text)
        self.timer = QTimer(self, singleShot=True)
        self.timer.timeout.connect(self.hide)

    def show_message(self, kind: str, message: str, detail: str = "", hold_ms: int = 1600) -> None:
        self.kind = kind
        self.message.setText(message)
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        self.adjustSize()
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        self.move(QPoint(area.center().x() - self.width() // 2, area.top() + 28))
        self.show()
        self.update()
        if hold_ms:
            self.timer.start(hold_ms)
        else:
            self.timer.stop()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        radius = min(bounds.height() / 2, 18)
        path.addRoundedRect(bounds, radius, radius)
        painter.fillPath(path, qcolor("rgba(22, 22, 26, 236)"))
        painter.setPen(QPen(qcolor(T.edge), 1))
        painter.drawPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(GLYPHS[self.kind]))
        painter.drawEllipse(QRectF(T.space_4 - 2, bounds.center().y() - 4, 8, 8))
