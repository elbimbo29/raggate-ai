"""Tests for the DeepEval judge.

Fast tests verify the wiring and error paths. Live tests (marked with
`@pytest.mark.live`) hit OpenAI and cost real money — run them
explicitly with `pytest -m live`, not in CI.
"""

import os

import pytest

from raggate.judge.deepeval_judge import DeepEvalJudge

# ---------- fast tests (no API calls) ----------

def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        DeepEvalJudge()


def test_name_includes_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-name-test")
    judge = DeepEvalJudge(model="gpt-4o-mini")
    assert judge.name == "deepeval-gpt-4o-mini"


# ---------- live tests (real API calls) ----------

@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_live_faithful_answer_scores_high():
    judge = DeepEvalJudge()
    result = judge.faithfulness(
        question="What is the rate limit?",
        answer="The rate limit is 100 requests per minute.",
        context=["The default rate limit is 100 requests per minute per API key."],
    )
    assert 0.5 <= result.score <= 1.0, f"expected high, got {result.score}: {result.reason}"
    assert result.latency_ms > 0


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_live_hallucinated_answer_scores_low():
    judge = DeepEvalJudge()
    result = judge.faithfulness(
        question="What is the rate limit?",
        answer="The rate limit is 10,000 requests per second and requires a paid enterprise plan.",
        context=["The default rate limit is 100 requests per minute per API key."],
    )
    assert result.score <= 0.5, f"expected low, got {result.score}: {result.reason}"