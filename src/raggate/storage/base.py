"""Storage interface for eval runs.

A RunStore persists a run's aggregate metrics and its per-case results
so the API can list, fetch, and compare them. Implementations can use
SQLite (default), Postgres, or anything else — callers only see this
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

RunKind = Literal["retrieval", "generation"]


@dataclass
class RunRecord:
    """Summary of a stored run — no per-case detail."""

    id: str
    kind: RunKind
    created_at: str
    retriever_name: str
    generator_name: str | None
    judge_name: str | None
    k: int
    n_cases: int
    cost_usd: float
    latency_ms: float
    metrics: dict[str, float]
    status: str = "succeeded"  # 'pending' | 'running' | 'succeeded' | 'failed'
    error: str | None = None

@dataclass
class CaseRecord:
    """One case's stored result. `payload` is the raw per-case dict."""

    case_id: str
    payload: dict


class RunStore(Protocol):
    """Persistence for eval runs."""

    def save_run(
        self,
        record: RunRecord,
        cases: list[CaseRecord],
    ) -> None:
        """Persist a run and its per-case results atomically."""
        ...

    def get_run(self, run_id: str) -> RunRecord | None:
        """Fetch a run's summary, or None if not found."""
        ...

    def get_run_cases(self, run_id: str) -> list[CaseRecord]:
        """Fetch every case result for a run, ordered by case_id."""
        ...

    def list_runs(
        self,
        kind: RunKind | None = None,
        limit: int = 20,
    ) -> list[RunRecord]:
        """List recent runs, newest first, optionally filtered by kind."""
        ...

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
    ) -> None: ...


    def delete_run(self, run_id: str) -> None:
        """Delete a run and its cases. No-op if missing."""
        ...