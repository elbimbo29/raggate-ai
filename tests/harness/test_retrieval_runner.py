"""Tests for the retrieval runner.

Uses a fake retriever so the test doesn't depend on the keyword baseline
or the real corpus. Every expected metric value is hand-computed.
"""

from raggate.dataset.models import Category, Difficulty, EvalCase
from raggate.harness.retrieval_runner import run_retrieval


class FakeRetriever:
    """Returns whatever chunks we tell it to, per question."""

    def __init__(self, mapping: dict[str, list[str]], name: str = "fake") -> None:
        self._mapping = mapping
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, question: str, k: int = 5) -> list[str]:
        return self._mapping.get(question, [])[:k]


def _case(id_: str, question: str, expected: list[str]) -> EvalCase:
    return EvalCase(
        id=id_,
        question=question,
        expected_answer="irrelevant for retrieval",
        expected_chunk_ids=expected,
        category=Category.MISC,
        difficulty=Difficulty.EASY,
    )


def test_runner_aggregates_metrics_across_cases():
    cases = [
        _case("c-1", "q1", ["A"]),
        _case("c-2", "q2", ["B"]),
    ]
    retriever = FakeRetriever({"q1": ["A", "X"], "q2": ["X", "Y"]})

    report = run_retrieval(cases, retriever, k=2)

    assert report.n_cases == 2
    assert report.retriever_name == "fake"
    assert report.k == 2

    # Case 1: expected {A}, retrieved [A, X]
    #   hit_rate@2 = 1.0
    #   mrr = 1/1 = 1.0
    #   recall@2 = 1/1 = 1.0
    #   context_precision: i=1 A relevant -> precision@1 = 1/1 = 1.0
    #     average over [1.0] = 1.0
    #   context_recall = 1.0
    # Case 2: expected {B}, retrieved [X, Y] -> all zeros
    # Averages: hit_rate=0.5, mrr=0.5, recall=0.5,
    #           context_precision=0.5, context_recall=0.5
    assert report.metrics["hit_rate@2"] == 0.5
    assert report.metrics["mrr"] == 0.5
    assert report.metrics["recall@2"] == 0.5
    assert report.metrics["context_precision"] == 0.5
    assert report.metrics["context_recall"] == 0.5


def test_runner_records_per_case_details():
    cases = [_case("c-1", "q1", ["A"])]
    retriever = FakeRetriever({"q1": ["A", "X"]})
    report = run_retrieval(cases, retriever, k=2)

    case = report.cases[0]
    assert case.case_id == "c-1"
    assert case.expected_chunk_ids == ["A"]
    assert case.retrieved_chunk_ids == ["A", "X"]


def test_runner_handles_empty_retrieval():
    cases = [_case("c-1", "q1", ["A"])]
    retriever = FakeRetriever({"q1": []})
    report = run_retrieval(cases, retriever, k=2)

    assert report.metrics["hit_rate@2"] == 0.0
    assert report.metrics["mrr"] == 0.0
    assert report.metrics["recall@2"] == 0.0
    assert report.metrics["context_precision"] == 0.0
    assert report.metrics["context_recall"] == 0.0


def test_report_serializes_to_json():
    cases = [_case("c-1", "q1", ["A"])]
    retriever = FakeRetriever({"q1": ["A"]})
    report = run_retrieval(cases, retriever, k=2)

    payload = report.model_dump_json()
    assert "retriever_name" in payload
    assert '"fake"' in payload
    assert "hit_rate@2" in payload