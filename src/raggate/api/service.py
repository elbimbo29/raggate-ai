"""Service layer: runs an evaluation and persists the result.

Keeps the endpoint handlers thin — they validate the request and
delegate here. The service knows about the golden set, the corpus,
the retrievers, generators, judges, and the store.
"""

from __future__ import annotations

from datetime import UTC, datetime

from raggate.dataset.loader import load_cases
from raggate.generator.template import TemplateGenerator
from raggate.harness.generation_runner import run_generation
from raggate.harness.retrieval_runner import run_retrieval
from raggate.judge.deepeval_judge import DeepEvalJudge
from raggate.judge.ragas_judge import RagasJudge
from raggate.retriever.chroma import ChromaRetriever
from raggate.retriever.corpus import load_corpus
from raggate.retriever.keyword import KeywordRetriever
from raggate.storage.base import CaseRecord, RunRecord, RunStore

DEFAULT_GOLDEN = "data/golden/golden.jsonl"
DEFAULT_CORPUS = "data/corpus/acmedb.jsonl"

RETRIEVERS = {"keyword": KeywordRetriever, "chroma": ChromaRetriever}
GENERATORS = {"template": TemplateGenerator}
JUDGES = {"deepeval": DeepEvalJudge, "ragas": RagasJudge}


class RunConfigError(ValueError):
    """Raised when the requested run config is invalid."""


def execute_run(
    store: RunStore,
    run_id: str,
    *,
    kind: str,
    retriever: str,
    generator: str | None,
    judge: str | None,
    k: int,
    golden_path: str | None,
    corpus_path: str | None,
) -> None:
    """Run the evaluation and persist the result.

    Called from a background task. Catches all exceptions and stores
    them on the run record so the client sees "failed" instead of a
    silent hang.
    """
    try:
        if retriever not in RETRIEVERS:
            raise RunConfigError(f"unknown retriever: {retriever!r}")
        if kind == "generation":
            if generator not in GENERATORS:
                raise RunConfigError(f"unknown or missing generator: {generator!r}")
            if judge not in JUDGES:
                raise RunConfigError(f"unknown or missing judge: {judge!r}")

        store.update_run(run_id, status="running")

        cases = load_cases(golden_path or DEFAULT_GOLDEN)
        corpus = load_corpus(corpus_path or DEFAULT_CORPUS)
        retriever_obj = RETRIEVERS[retriever](corpus)

        if kind == "retrieval":
            report = run_retrieval(cases, retriever_obj, k=k)
            record = RunRecord(
                id=run_id,
                kind="retrieval",
                created_at=datetime.now(UTC).isoformat(),
                retriever_name=report.retriever_name,
                generator_name=None,
                judge_name=None,
                k=report.k,
                n_cases=report.n_cases,
                cost_usd=0.0,
                latency_ms=0.0,
                metrics=report.metrics,
                status="succeeded",
                error=None,
            )
            case_records = [
                CaseRecord(case_id=c.case_id, payload=c.model_dump())
                for c in report.cases
            ]
        else:
            generator_obj = GENERATORS[generator]()
            judge_obj = JUDGES[judge]()
            report = run_generation(
                cases, corpus, retriever_obj, generator_obj, judge_obj, k=k
            )
            record = RunRecord(
                id=run_id,
                kind="generation",
                created_at=datetime.now(UTC).isoformat(),
                retriever_name=report.retriever_name,
                generator_name=report.generator_name,
                judge_name=report.judge_name,
                k=report.k,
                n_cases=report.n_cases,
                cost_usd=report.cost_usd_total,
                latency_ms=report.latency_ms_total,
                metrics=report.metrics,
                status="succeeded",
                error=None,
            )
            case_records = [
                CaseRecord(case_id=c.case_id, payload=c.model_dump())
                for c in report.cases
            ]

        # The run row was inserted with placeholder metrics by the endpoint.
        # Replace it with the completed record and add the case rows.
        # We do a delete + insert to reuse save_run's atomicity.
        _replace_run(store, record, case_records)

    except Exception as e:
        store.update_run(run_id, status="failed", error=str(e))


def _replace_run(
    store: RunStore,
    record: RunRecord,
    cases: list[CaseRecord],
) -> None:
    store.delete_run(record.id)
    store.save_run(record, cases)