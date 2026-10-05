"""Request and response schemas for the eval API.

Separates the HTTP contract from the internal report models
(RetrievalReport, GenerationReport). The report models stay stable;
the API schemas are what clients see, and can evolve independently.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RunKind = Literal["retrieval", "generation"]
RunStatus = Literal["pending", "running", "succeeded", "failed"]


# ---------- requests ----------

class RunRequest(BaseModel):
    """Body for POST /eval/run."""

    kind: RunKind = Field(
        ...,
        description="Which evaluation to run.",
    )
    retriever: str = Field(
        "chroma",
        description="Retriever key: 'keyword' or 'chroma'.",
    )
    generator: str | None = Field(
        None,
        description="Generator key (required when kind='generation').",
    )
    judge: str | None = Field(
        None,
        description="Judge key: 'deepeval' or 'ragas' (required when kind='generation').",
    )
    k: int = Field(5, ge=1, le=20, description="Top-k for retrieval.")
    golden_path: str | None = Field(
        None,
        description="Override the default golden set path.",
    )
    corpus_path: str | None = Field(
        None,
        description="Override the default corpus path.",
    )


class CompareRequest(BaseModel):
    """Body for POST /eval/compare."""

    baseline_run_id: str = Field(..., min_length=1)
    candidate_run_id: str = Field(..., min_length=1)


# ---------- responses ----------

class RunResponse(BaseModel):
    """Response for POST /eval/run."""

    run_id: str
    status: RunStatus


class RunSummary(BaseModel):
    """One row in GET /eval/runs."""

    id: str
    kind: RunKind
    status: RunStatus
    created_at: str
    retriever_name: str
    generator_name: str | None
    judge_name: str | None
    k: int
    n_cases: int
    cost_usd: float
    latency_ms: float
    metrics: dict[str, float]
    error: str | None = None


class RunDetail(BaseModel):
    """Response for GET /eval/runs/{id} — summary plus per-case results."""

    summary: RunSummary
    cases: list[dict] = Field(
        ...,
        description="Per-case results, shape depends on run kind.",
    )


class CompareResponse(BaseModel):
    """Response for POST /eval/compare."""

    baseline_run_id: str
    candidate_run_id: str
    kind: RunKind
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    deltas: dict[str, float] = Field(
        ...,
        description="candidate - baseline, for each metric present in either run.",
    )