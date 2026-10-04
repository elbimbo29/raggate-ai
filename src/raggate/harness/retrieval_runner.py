"""Retrieval runner.

Takes a golden set and a retriever, runs every case, computes all five
retrieval metrics per case, and aggregates. Produces a RetrievalReport
that serializes cleanly to JSON.

The runner knows nothing about specific retrievers or specific metrics
implementations — it just calls retrieve() per case and passes the
result to each metric function. That keeps it easy to test with fakes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from raggate.dataset.models import EvalCase
from raggate.metrics.context import context_precision, context_recall
from raggate.metrics.retrieval import hit_rate_at_k, mrr, recall_at_k
from raggate.retriever.base import Retriever


class CaseResult(BaseModel):
    """One golden case's retrieval outcome and scores."""

    case_id: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    metrics: dict[str, float]


class RetrievalReport(BaseModel):
    """Aggregate of a full retrieval run."""

    retriever_name: str
    k: int
    n_cases: int
    timestamp: str
    metrics: dict[str, float] = Field(
        ..., description="Mean of each metric across all cases."
    )
    cases: list[CaseResult]


def _score_case(
    expected: set[str], retrieved: list[str], k: int
) -> dict[str, float]:
    return {
        f"hit_rate@{k}": hit_rate_at_k(expected, retrieved, k),
        "mrr": mrr(expected, retrieved),
        f"recall@{k}": recall_at_k(expected, retrieved, k),
        "context_precision": context_precision(expected, retrieved, k),
        "context_recall": context_recall(expected, retrieved, k),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_retrieval(
    cases: list[EvalCase],
    retriever: Retriever,
    k: int = 5,
) -> RetrievalReport:
    """Run every case through the retriever and score the results."""
    case_results: list[CaseResult] = []

    for case in cases:
        expected = set(case.expected_chunk_ids)
        retrieved = retriever.retrieve(case.question, k=k)
        case_results.append(
            CaseResult(
                case_id=case.id,
                expected_chunk_ids=case.expected_chunk_ids,
                retrieved_chunk_ids=retrieved,
                metrics=_score_case(expected, retrieved, k),
            )
        )

    # Average each metric across all cases. The metric names are stable
    # (they come from _score_case), so use the first case's keys.
    metric_names = list(case_results[0].metrics.keys())
    aggregate = {
        name: _mean([c.metrics[name] for c in case_results])
        for name in metric_names
    }

    return RetrievalReport(
        retriever_name=retriever.name,
        k=k,
        n_cases=len(case_results),
        timestamp=datetime.now(UTC).isoformat(),
        metrics=aggregate,
        cases=case_results,
    )