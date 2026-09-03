"""Batched SQLite cache for high-volume study collection."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype

from reasonese.cache import trace_to_dict, traces_from_dicts
from reasonese.conversation import ConversationTrace
from reasonese.judging import Judgment
from reasonese.judgment_cache import judgment_to_dict, judgments_from_dicts
from reasonese.study import Trial, TrialId


def _encode(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _decode(raw: object, *, record: str) -> object:
    if not isinstance(raw, str):
        raise ValueError(f"cached {record} payload must be text")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"cached {record} payload is not valid JSON") from error


@beartype
@dataclass(frozen=True, slots=True)
class SqliteStudyCache:
    """Conversation traces and judgments keyed by stable trial identifiers."""

    path: Path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS traces ("
                "trial_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS judgments ("
                "trial_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            yield connection
            connection.commit()
        finally:
            connection.close()

    def load_traces(self, trials: tuple[Trial, ...]) -> dict[TrialId, ConversationTrace]:
        """Load known trial traces in one query, reusing their validated matchups."""
        with self._connection() as connection:
            rows = connection.execute("SELECT trial_id, payload FROM traces").fetchall()
        payloads = dict(rows)
        selected = tuple(
            (trial, _decode(payloads[str(trial.trial_id)], record="trace"))
            for trial in trials
            if str(trial.trial_id) in payloads
        )
        traces = traces_from_dicts(
            tuple(raw for _, raw in selected),
            tuple(trial.matchup for trial, _ in selected),
        )
        return {
            trial.trial_id: trace
            for (trial, _), trace in zip(selected, traces, strict=True)
        }

    @beartype
    def put_traces(self, traces: tuple[tuple[TrialId, ConversationTrace], ...]) -> None:
        """Insert or replace traces in one transaction."""
        if not traces:
            return
        with self._connection() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO traces (trial_id, payload) VALUES (?, ?)",
                tuple((str(trial_id), _encode(trace_to_dict(trace))) for trial_id, trace in traces),
            )

    def load_judgments(self, trials: tuple[Trial, ...]) -> dict[TrialId, Judgment]:
        """Load known trial judgments in one query, reusing their validated matchups."""
        with self._connection() as connection:
            rows = connection.execute("SELECT trial_id, payload FROM judgments").fetchall()
        payloads = dict(rows)
        selected = tuple(
            (trial, _decode(payloads[str(trial.trial_id)], record="judgment"))
            for trial in trials
            if str(trial.trial_id) in payloads
        )
        judgments = judgments_from_dicts(
            tuple(raw for _, raw in selected),
            tuple(trial.matchup for trial, _ in selected),
        )
        return {
            trial.trial_id: judgment
            for (trial, _), judgment in zip(selected, judgments, strict=True)
        }

    @beartype
    def put_judgments(self, judgments: tuple[tuple[TrialId, Judgment], ...]) -> None:
        """Insert or replace judgments in one transaction."""
        if not judgments:
            return
        with self._connection() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO judgments (trial_id, payload) VALUES (?, ?)",
                tuple(
                    (str(trial_id), _encode(judgment_to_dict(judgment)))
                    for trial_id, judgment in judgments
                ),
            )
