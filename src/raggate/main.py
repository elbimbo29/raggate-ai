"""FastAPI entrypoint for RAGGate AI."""

from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query

from raggate import __version__
from raggate.api.schemas import (
    CompareRequest,
    CompareResponse,
    GateRequest,
    GateResponse,
    MetricDeltaResponse,
    RunDetail,
    RunRequest,
    RunResponse,
    RunSummary,
)
from raggate.api.service import execute_run
from raggate.config import settings
from raggate.gate.evaluator import evaluate_gate
from raggate.gate.models import Thresholds
from raggate.storage.base import RunRecord
from raggate.storage.sqlite import SQLiteRunStore, new_run_id

app = FastAPI(
    title="RAGGate AI",
    description="Automated evaluation harness and regression gate for RAG pipelines.",
    version=__version__,
)

# Single store instance attached to app state. Tests override this.
_store: SQLiteRunStore | None = None


def get_store() -> SQLiteRunStore:
    """Lazily create the store so tests can override before first use."""
    global _store
    if _store is None:
        Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
        _store = SQLiteRunStore(settings.db_path)
    return _store


def set_store(store: SQLiteRunStore | None) -> None:
    """Test hook — swap in an in-memory store, or reset with None."""
    global _store
    _store = store


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "env": settings.env,
    }


@app.post("/eval/run", response_model=RunResponse, status_code=202)
def start_run(req: RunRequest, background: BackgroundTasks) -> RunResponse:
    """Kick off an evaluation run. Returns immediately with a run ID."""
    if req.kind == "generation" and (req.generator is None or req.judge is None):
        raise HTTPException(
            status_code=400,
            detail="kind='generation' requires both 'generator' and 'judge'.",
        )

    store = get_store()
    run_id = new_run_id()

    placeholder = RunRecord(
        id=run_id,
        kind=req.kind,
        created_at=datetime.now(UTC).isoformat(),
        retriever_name=req.retriever,
        generator_name=req.generator,
        judge_name=req.judge,
        k=req.k,
        n_cases=0,
        cost_usd=0.0,
        latency_ms=0.0,
        metrics={},
        status="pending",
        error=None,
    )
    store.save_run(placeholder, [])

    background.add_task(
        execute_run,
        store,
        run_id,
        kind=req.kind,
        retriever=req.retriever,
        generator=req.generator,
        judge=req.judge,
        k=req.k,
        golden_path=req.golden_path,
        corpus_path=req.corpus_path,
    )

    return RunResponse(run_id=run_id, status="pending")


@app.get("/eval/runs", response_model=list[RunSummary])
def list_runs(
    kind: str | None = Query(
        None, description="Filter by 'retrieval' or 'generation'."
    ),
    limit: int = Query(20, ge=1, le=100, description="Max runs to return."),
) -> list[RunSummary]:
    """List recent runs, newest first, optionally filtered by kind."""
    store = get_store()
    records = store.list_runs(kind=kind, limit=limit)  # type: ignore[arg-type]
    return [_record_to_summary(r) for r in records]


@app.get("/eval/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str) -> RunDetail:
    """Fetch a run's summary plus its per-case results."""
    store = get_store()
    record = store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

    cases = store.get_run_cases(run_id)
    return RunDetail(
        summary=_record_to_summary(record),
        cases=[c.payload for c in cases],
    )


@app.post("/eval/compare", response_model=CompareResponse)
def compare_runs(req: CompareRequest) -> CompareResponse:
    """Compare two runs' metrics, return per-metric deltas."""
    store = get_store()

    baseline = store.get_run(req.baseline_run_id)
    if baseline is None:
        raise HTTPException(
            status_code=404,
            detail=f"baseline run not found: {req.baseline_run_id}",
        )

    candidate = store.get_run(req.candidate_run_id)
    if candidate is None:
        raise HTTPException(
            status_code=404,
            detail=f"candidate run not found: {req.candidate_run_id}",
        )

    if baseline.kind != candidate.kind:
        raise HTTPException(
            status_code=400,
            detail=(
                f"cannot compare different run kinds: "
                f"{baseline.kind} vs {candidate.kind}"
            ),
        )

    all_keys = set(baseline.metrics) | set(candidate.metrics)
    deltas = {
        k: candidate.metrics.get(k, 0.0) - baseline.metrics.get(k, 0.0)
        for k in all_keys
    }

    return CompareResponse(
        baseline_run_id=baseline.id,
        candidate_run_id=candidate.id,
        kind=baseline.kind,  # type: ignore[arg-type]
        baseline_metrics=baseline.metrics,
        candidate_metrics=candidate.metrics,
        deltas=deltas,
    )


@app.post("/eval/gate", response_model=GateResponse)
def gate_runs(req: GateRequest) -> GateResponse:
    """Compare two runs and return a pass/fail verdict.

    Refuses to gate a run that isn't finished — comparing against a
    placeholder record would silently produce nonsense (all-zero
    baseline, everything looks like an improvement).
    """
    store = get_store()

    baseline = store.get_run(req.baseline_run_id)
    if baseline is None:
        raise HTTPException(
            status_code=404,
            detail=f"baseline run not found: {req.baseline_run_id}",
        )

    candidate = store.get_run(req.candidate_run_id)
    if candidate is None:
        raise HTTPException(
            status_code=404,
            detail=f"candidate run not found: {req.candidate_run_id}",
        )

    if baseline.kind != candidate.kind:
        raise HTTPException(
            status_code=400,
            detail=(
                f"cannot gate different run kinds: "
                f"{baseline.kind} vs {candidate.kind}"
            ),
        )

    if baseline.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"baseline run is not finished: status={baseline.status!r}",
        )
    if candidate.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"candidate run is not finished: status={candidate.status!r}",
        )

    result = evaluate_gate(
        baseline_metrics=baseline.metrics,
        candidate_metrics=candidate.metrics,
        thresholds=Thresholds(per_metric=req.thresholds),
    )

    return GateResponse(
        passed=result.passed,
        baseline_run_id=baseline.id,
        candidate_run_id=candidate.id,
        kind=baseline.kind,  # type: ignore[arg-type]
        regressions=[
            MetricDeltaResponse(**r.model_dump()) for r in result.regressions
        ],
        improvements=[
            MetricDeltaResponse(**r.model_dump()) for r in result.improvements
        ],
        unchanged=result.unchanged,
    )


def _record_to_summary(record: RunRecord) -> RunSummary:
    return RunSummary(
        id=record.id,
        kind=record.kind,  # type: ignore[arg-type]
        status=record.status,  # type: ignore[arg-type]
        created_at=record.created_at,
        retriever_name=record.retriever_name,
        generator_name=record.generator_name,
        judge_name=record.judge_name,
        k=record.k,
        n_cases=record.n_cases,
        cost_usd=record.cost_usd,
        latency_ms=record.latency_ms,
        metrics=record.metrics,
        error=record.error,
    )