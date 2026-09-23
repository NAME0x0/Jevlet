"""Local, explicit desktop feedback. Screenshots and typed content are never persisted."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class DesktopFeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS desktop_feedback (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    task TEXT NOT NULL,
                    suggestion TEXT NOT NULL,
                    action_kind TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating IN (-1, 1))
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def rate(self, task: str, suggestion: str, action_kind: str, rating: int) -> None:
        if rating not in (-1, 1):
            raise ValueError("Rating must be thumbs up or thumbs down")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO desktop_feedback (task, suggestion, action_kind, rating) "
                "VALUES (?, ?, ?, ?)",
                (task[:500], suggestion[:500], action_kind[:40], rating),
            )
