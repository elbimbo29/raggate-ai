"""Tests for the generation runner.

Uses FakeJudge + TemplateGenerator so no API calls are made. Every
expected score is hand-computed from the fake judge's word-overlap
heuristics.
"""

from raggate.dataset.models import Category, Difficulty, EvalCase
from raggate.generator.template import TemplateGenerator
from raggate.harness.generation_runner import run_generation
from raggate.judge.fake import FakeJudge
from raggate.retriever.corpus import Chunk


class FakeRetriever:
    def __init__(self, mapping: dict[str, list[str]], name: str = "fake-ret") -> None:
        self._mapping = mapping
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, question: str, k: int = 5) -> list[str]:
        return self._mapping.get(question, [])[:k]


def _case(id_: str, question: str, expected: str) -> EvalCase:
    return EvalCase(
        id=id_,
        question=question,
        expected_answer=expected,
        expected_chunk_ids=["c1"],
        category=Category.MISC,
        difficulty=Difficulty.EASY,
    )


def _corpus() -> list[Chunk]:
    return [
        Chunk(chunk_id="c1", section="a", text="rate limit is 100 per minute"),
    ]


def test_generation_runner_produces_expected_shape():
    cases = [_case("c-1", "what is the rate limit", "rate limit is 100")]
    retriever = FakeRetriever({"what is the rate limit": ["c1"]})
    generator = TemplateGenerator()
    judge = FakeJudge()

    report = run_generation(cases, _corpus(), retriever, generator, judge, k=1)

    assert report.n_cases == 1
    assert report.retriever_name == "fake-ret"
    assert report.generator_name == "template-v1"
    assert report.judge_name == "fake-judge"
    assert set(report.metrics.keys()) == {
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    }
    # All judge scores should be between 0 and 1.
    for v in report.metrics.values():
        assert 0.0 <= v <= 1.0


def test_template_answer_is_copied_from_context():
    cases = [_case("c-1", "what is the rate limit", "rate limit is 100")]
    retriever = FakeRetriever({"what is the rate limit": ["c1"]})
    generator = TemplateGenerator()
    judge = FakeJudge()

    report = run_generation(cases, _corpus(), retriever, generator, judge, k=1)

    case = report.cases[0]
    # Template generator copies context verbatim -> answer == context text.
    assert case.answer == "rate limit is 100 per minute"
    # Faithfulness should be 1.0 — every answer word is in the context.
    assert case.metrics["faithfulness"] == 1.0


def test_cost_and_latency_are_tracked():
    cases = [_case("c-1", "what is the rate limit", "rate limit is 100")]
    retriever = FakeRetriever({"what is the rate limit": ["c1"]})
    report = run_generation(
        cases, _corpus(), retriever, TemplateGenerator(), FakeJudge(), k=1
    )
    # FakeJudge returns cost=0 and latency=0.
    assert report.cost_usd_total == 0.0
    assert report.latency_ms_total == 0.0


def test_empty_retrieval_still_produces_report():
    cases = [_case("c-1", "what is the rate limit", "rate limit is 100")]
    retriever = FakeRetriever({"what is the rate limit": []})
    report = run_generation(
        cases, _corpus(), retriever, TemplateGenerator(), FakeJudge(), k=1
    )
    case = report.cases[0]
    assert case.retrieved_chunk_ids == []
    # Template generator returns "" for empty context.
    assert case.answer == ""
    # Faithfulness of an empty answer is 0.0.
    assert case.metrics["faithfulness"] == 0.0