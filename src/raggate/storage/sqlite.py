"""SQLite implementation of RunStore.

Uses the stdlib sqlite3 module with a single connection per store.
For a portfolio project with a single process this is fine; a
production deployment would use a connection pool or async driver.

The schema is created on first use (CREATE TABLE IF NOT EXISTS), so
no migration tool is needed yet.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from raggate.storage.base import CaseRecord, RunKind, RunRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    retriever_name  TEXT NOT NULL,
    generator_name  TEXT,
    judge_name      TEXT,
    k               INTEGER NOT NULL,
    n_cases         INTEGER NOT NULL,
    cost_usd        REAL NOT NULL,
    latency_ms      REAL NOT NULL,
    metrics         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'succeeded',
    error           TEXT
);

CREATE TABLE IF NOT EXISTS cases (
    run_id   TEXT NOT NULL,
    case_id  TEXT NOT NULL,
    payload  TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_kind_created
    ON runs(kind, created_at DESC);
"""


def new_run_id() -> str:
    return uuid.uuid4().hex


class SQLiteRunStore:
    """RunStore implementation backed by a SQLite file or ':memory:'."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        # check_same_thread=False because FastAPI background tasks may
        # touch the connection from a worker thread.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save_run(
        self,
        record: RunRecord,
        cases: list[CaseRecord],
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO runs (
                    id, kind, created_at, retriever_name, generator_name,
                    judge_name, k, n_cases, cost_usd, latency_ms, metrics,
                    status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.kind,
                    record.created_at,
                    record.retriever_name,
                    record.generator_name,
                    record.judge_name,
                    record.k,
                    record.n_cases,
                    record.cost_usd,
                    record.latency_ms,
                    json.dumps(record.metrics),
                    record.status,
                    record.error,
                ),
            )
            self._conn.executemany(
                "INSERT INTO cases (run_id, case_id, payload) VALUES (?, ?, ?)",
                [
                    (record.id, c.case_id, json.dumps(c.payload))
                    for c in cases
                ],
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def get_run_cases(self, run_id: str) -> list[CaseRecord]:
        rows = self._conn.execute(
            "SELECT case_id, payload FROM cases WHERE run_id = ? ORDER BY case_id",
            (run_id,),
        ).fetchall()
        return [
            CaseRecord(case_id=r["case_id"], payload=json.loads(r["payload"]))
            for r in rows
        ]

    def list_runs(
        self,
        kind: RunKind | None = None,
        limit: int = 20,
    ) -> list[RunRecord]:
        if kind is None:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        error: str | None = None,
        metrics: dict[str, float] | None = None,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        n_cases: int | None = None,
    ) -> None:
        """Patch selected fields on a run. No-op for None fields."""
        sets: list[str] = []
        values: list = []
        if status is not None:
            sets.append("status = ?")
            values.append(status)
        if error is not None:
            sets.append("error = ?")
            values.append(error)
        if metrics is not None:
            sets.append("metrics = ?")
            values.append(json.dumps(metrics))
        if cost_usd is not None:
            sets.append("cost_usd = ?")
            values.append(cost_usd)
        if latency_ms is not None:
            sets.append("latency_ms = ?")
            values.append(latency_ms)
        if n_cases is not None:
            sets.append("n_cases = ?")
            values.append(n_cases)
        if not sets:
            return
        values.append(run_id)
        with self._conn:
            self._conn.execute(
                f"UPDATE runs SET {', '.join(sets)} WHERE id = ?",
                values,
            )

    def delete_run(self, run_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM cases WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def close(self) -> None:
        self._conn.close()


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        kind=row["kind"],
        created_at=row["created_at"],
        retriever_name=row["retriever_name"],
        generator_name=row["generator_name"],
        judge_name=row["judge_name"],
        k=row["k"],
        n_cases=row["n_cases"],
        cost_usd=row["cost_usd"],
        latency_ms=row["latency_ms"],
        metrics=json.loads(row["metrics"]),
        status=row["status"],
        error=row["error"],
    )