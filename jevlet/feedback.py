"""Local, explicit labels for improving the daily-driver router."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LabeledDecision:
    id: int
    state: str
    question: str
    options: tuple[tuple[str, str], ...]
    label: str


class FeedbackStore:
    """Feedback is local-only; uncorrected thumbs-down is never a training label."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    state TEXT NOT NULL,
                    question TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    selected TEXT NOT NULL,
                    approved INTEGER,
                    corrected_option TEXT,
                    rated_at TEXT
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def log_decision(
        self,
        state: str,
        question: str,
        options: tuple[tuple[str, str], ...],
        selected: str,
    ) -> int:
        names = [name for name, _ in options]
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("decisions need at least two uniquely named options")
        if selected not in names:
            raise ValueError("selected must be one of the options")
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO decisions(state, question, options_json, selected)
                   VALUES (?, ?, ?, ?)""",
                (state, question, json.dumps(options, ensure_ascii=False), selected),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def submit_feedback(
        self, decision_id: int, *, approved: bool, corrected_option: str | None = None
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT options_json, selected, approved FROM decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown decision ID: {decision_id}")
            if row["approved"] is not None:
                raise ValueError("feedback has already been recorded for this decision")
            if approved and corrected_option is not None:
                raise ValueError("an approved decision cannot have a correction")
            names = [name for name, _ in json.loads(row["options_json"])]
            if corrected_option is not None and corrected_option not in names:
                raise ValueError("correction must name an existing option")
            if not approved and corrected_option == row["selected"]:
                raise ValueError("correction must differ from the rejected selection")
            connection.execute(
                """UPDATE decisions SET approved = ?, corrected_option = ?,
                   rated_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (int(approved), corrected_option, decision_id),
            )

    def labeled_decisions(self) -> list[LabeledDecision]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, state, question, options_json, selected, approved, corrected_option
                   FROM decisions WHERE approved = 1 OR corrected_option IS NOT NULL
                   ORDER BY id"""
            ).fetchall()
        return [
            LabeledDecision(
                id=row["id"],
                state=row["state"],
                question=row["question"],
                options=tuple(tuple(pair) for pair in json.loads(row["options_json"])),
                label=row["selected"] if row["approved"] else row["corrected_option"],
            )
            for row in rows
        ]
