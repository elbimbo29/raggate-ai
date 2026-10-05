"""FastAPI entrypoint for RAGGate AI."""

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException

from raggate import __version__
from raggate.api.schemas import RunRequest, RunResponse
from raggate.api.service import execute_run
from raggate.config import settings
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


def set_store(store: SQLiteRunStore) -> None:
    """Test hook — swap in an in-memory store."""
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

    # Insert a placeholder run so GET /eval/runs/{id} returns something
    # immediately after this call returns.
    from datetime import UTC, datetime

    from raggate.storage.base import RunRecord

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