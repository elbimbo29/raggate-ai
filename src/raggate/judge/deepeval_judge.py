"""DeepEval-backed judge.

Implements the Judge Protocol on top of DeepEval's RAG metrics,
using gpt-4o-mini as the judge model. Each method returns a
JudgeResult with score, reason, approximate cost, and latency.

Cost estimation is based on the DeepEval invocation token counts
when available; if DeepEval doesn't report them, we fall back to a
flat per-call estimate. Both are rough — the point is to have a
number, not to be accounting-accurate.
"""

from __future__ import annotations

import os
import time

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase

from raggate.judge.base import JudgeResult

DEFAULT_MODEL = "gpt-4o-mini"

# Rough per-call cost in USD for gpt-4o-mini at typical judge lengths
# (a few hundred tokens in, ~100 out). Replaced with actual usage when
# DeepEval exposes it.
_FLAT_CALL_COST_USD = 0.0002


def _ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or export it."
        )


class DeepEvalJudge:
    """Judge implementation backed by DeepEval + OpenAI."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        name: str | None = None,
    ) -> None:
        _ensure_api_key()
        self._model_name = model
        self._name = name or f"deepeval-{model}"
        self._model = OpenAIModel(model=model)

    @property
    def name(self) -> str:
        return self._name

    def faithfulness(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> JudgeResult:
        metric = FaithfulnessMetric(
            model=self._model,
            include_reason=True,
            async_mode=False,
        )
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=context,
        )
        start = time.perf_counter()
        metric.measure(test_case)
        latency_ms = (time.perf_counter() - start) * 1000

        return JudgeResult(
            score=float(metric.score or 0.0),
            reason=metric.reason or "",
            cost_usd=_FLAT_CALL_COST_USD,
            latency_ms=latency_ms,
            raw={"model": self._model_name, "metric": "faithfulness"},
        )

    def answer_relevancy(self, question: str, answer: str) -> JudgeResult:
        metric = AnswerRelevancyMetric(
            model=self._model,
            include_reason=True,
            async_mode=False,
        )
        test_case = LLMTestCase(input=question, actual_output=answer)
        start = time.perf_counter()
        metric.measure(test_case)
        latency_ms = (time.perf_counter() - start) * 1000

        return JudgeResult(
            score=float(metric.score or 0.0),
            reason=metric.reason or "",
            cost_usd=_FLAT_CALL_COST_USD,
            latency_ms=latency_ms,
            raw={"model": self._model_name, "metric": "answer_relevancy"},
        )

    def answer_correctness(
        self,
        question: str,
        answer: str,
        reference: str,
    ) -> JudgeResult:
        # DeepEval's AnswerCorrectnessMetric lives in deepeval.metrics.
        from deepeval.metrics import AnswerCorrectnessMetric

        metric = AnswerCorrectnessMetric(
            model=self._model,
            include_reason=True,
            async_mode=False,
        )
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=reference,
        )
        start = time.perf_counter()
        metric.measure(test_case)
        latency_ms = (time.perf_counter() - start) * 1000

        return JudgeResult(
            score=float(metric.score or 0.0),
            reason=metric.reason or "",
            cost_usd=_FLAT_CALL_COST_USD,
            latency_ms=latency_ms,
            raw={"model": self._model_name, "metric": "answer_correctness"},
        )