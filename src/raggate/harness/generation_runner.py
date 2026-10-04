"""Generation runner.

Full pipeline: for each golden case, retrieve context, generate an
answer, judge the answer on three metrics (faithfulness, answer
relevancy, answer correctness), and aggregate.

Cost and latency are tracked per case and in total — a generation run
costs real money and takes real time, unlike retrieval.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from raggate.dataset.models import EvalCase
from raggate.generator.base import Generator
from raggate.judge.base import Judge, JudgeResult
from raggate.retriever.base import Retriever
from raggate.retriever.corpus import Chunk


class GenerationCaseResult(BaseModel):
    """One case's generation outcome and judge scores."""

    case_id: str
    question: str
    retrieved_chunk_ids: list[str]
    answer: str
    reference_answer: str
    metrics: dict[str, float]
    reasons: dict[str, str]
    cost_usd: float
    latency_ms: float


class GenerationReport(BaseModel):
    """Aggregate of a full generation run."""

    retriever_name: str
    generator_name: str
    judge_name: str
    k: int
    n_cases: int
    timestamp: str
    metrics: dict[str, float] = Field(
        ..., description="Mean of each generation metric across all cases."
    )
    cost_usd_total: float
    latency_ms_total: float
    cases: list[GenerationCaseResult]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score_answer(
    judge: Judge,
    question: str,
    answer: str,
    context_texts: list[str],
    reference: str,
) -> tuple[dict[str, float], dict[str, str], float, float]:
    """Run all three judge methods. Returns metrics, reasons, cost, latency."""
    faith: JudgeResult = judge.faithfulness(question, answer, context_texts)
    relev: JudgeResult = judge.answer_relevancy(question, answer)
    correct: JudgeResult = judge.answer_correctness(question, answer, reference)

    metrics = {
        "faithfulness": faith.score,
        "answer_relevancy": relev.score,
        "answer_correctness": correct.score,
    }
    reasons = {
        "faithfulness": faith.reason,
        "answer_relevancy": relev.reason,
        "answer_correctness": correct.reason,
    }
    cost = faith.cost_usd + relev.cost_usd + correct.cost_usd
    latency = faith.latency_ms + relev.latency_ms + correct.latency_ms
    return metrics, reasons, cost, latency


def run_generation(
    cases: list[EvalCase],
    corpus: list[Chunk],
    retriever: Retriever,
    generator: Generator,
    judge: Judge,
    k: int = 5,
) -> GenerationReport:
    """Run every case through retrieve → generate → judge.

    The corpus is passed explicitly so the runner can resolve retrieved
    chunk IDs to text before generation. Retrievers only return IDs;
    something has to know the text, and passing it here keeps the
    Retriever interface minimal.
    """
    chunk_lookup = {c.chunk_id: c for c in corpus}
    case_results: list[GenerationCaseResult] = []

    for case in cases:
        retrieved_ids = retriever.retrieve(case.question, k=k)
        context_chunks = [
            chunk_lookup[cid] for cid in retrieved_ids if cid in chunk_lookup
        ]
        context_texts = [c.text for c in context_chunks]

        answer = generator.generate(case.question, context_chunks)

        metrics, reasons, cost, latency = _score_answer(
            judge=judge,
            question=case.question,
            answer=answer,
            context_texts=context_texts,
            reference=case.expected_answer,
        )

        case_results.append(
            GenerationCaseResult(
                case_id=case.id,
                question=case.question,
                retrieved_chunk_ids=retrieved_ids,
                answer=answer,
                reference_answer=case.expected_answer,
                metrics=metrics,
                reasons=reasons,
                cost_usd=cost,
                latency_ms=latency,
            )
        )

    metric_names = list(case_results[0].metrics.keys())
    aggregate = {
        name: _mean([c.metrics[name] for c in case_results]) for name in metric_names
    }

    return GenerationReport(
        retriever_name=retriever.name,
        generator_name=generator.name,
        judge_name=judge.name,
        k=k,
        n_cases=len(case_results),
        timestamp=datetime.now(UTC).isoformat(),
        metrics=aggregate,
        cost_usd_total=sum(c.cost_usd for c in case_results),
        latency_ms_total=sum(c.latency_ms for c in case_results),
        cases=case_results,
    )

def _build_chunk_lookup(retriever: Retriever) -> dict[str, Chunk]:
    """Best-effort chunk ID -> Chunk lookup from the retriever.

    Both KeywordRetriever and ChromaRetriever store their chunks in a
    private `_chunks` attribute. We use it here rather than adding a
    public accessor to every retriever — keeps the Retriever interface
    narrow. If a future retriever doesn't expose chunks, this raises
    with a clear message.
    """
    chunks = getattr(retriever, "_chunks", None)
    if chunks is None:
        raise TypeError(
            f"Retriever {retriever.name!r} does not expose chunks for context lookup. "
            "The retriever must store its corpus in a `_chunks` attribute, or the "
            "harness needs a different way to resolve chunk IDs to text."
        )
    return {c.chunk_id: c for c in chunks}