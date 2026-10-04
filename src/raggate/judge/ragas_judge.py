"""RAGAS-backed judge.

Implements the Judge Protocol using RAGAS 0.4's collections metrics.
Uses an explicit AsyncOpenAI client (RAGAS 0.4 removed text-only
llm_factory).

RAGAS 0.4 has two collections metrics that map cleanly to our
Judge Protocol: Faithfulness and AnswerRelevancy. There is no direct
"answer correctness" metric in the collections API, so this judge
implements only faithfulness and answer_relevancy. The cross-check
runner compares those two against DeepEval.
"""

from __future__ import annotations

import asyncio
import os
import time

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from raggate.judge.base import JudgeResult

DEFAULT_MODEL = "gpt-4o-mini"
_FLAT_CALL_COST_USD = 0.0002


def _ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or export it."
        )


def _run_async(coro):
    """Bridge RAGAS's async API to our sync harness.

    RAGAS collections metrics are async-only. The harness is sync, so
    we spin up a fresh event loop per call. Slower than a shared loop,
    but matches the shape of DeepEvalJudge.
    """
    return asyncio.run(coro)


class RagasJudge:
    """Judge implementation backed by RAGAS 0.4."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        name: str | None = None,
    ) -> None:
        _ensure_api_key()
        self._model_name = model
        self._name = name or f"ragas-{model}"
        client = AsyncOpenAI(timeout=60.0, max_retries=2)
        self._llm = llm_factory(model, client=client)

    @property
    def name(self) -> str:
        return self._name

    def faithfulness(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> JudgeResult:
        metric = Faithfulness(llm=self._llm)
        start = time.perf_counter()
        result = _run_async(
            metric.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=context,
            )
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return JudgeResult(
            score=float(result.value),
            reason=str(getattr(result, "reason", "") or ""),
            cost_usd=_FLAT_CALL_COST_USD,
            latency_ms=latency_ms,
            raw={
                "model": self._model_name,
                "metric": "faithfulness",
                "backend": "ragas",
            },
        )

    def answer_relevancy(self, question: str, answer: str) -> JudgeResult:
        metric = AnswerRelevancy(llm=self._llm)
        start = time.perf_counter()
        result = _run_async(
            metric.ascore(
                user_input=question,
                response=answer,
            )
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return JudgeResult(
            score=float(result.value),
            reason=str(getattr(result, "reason", "") or ""),
            cost_usd=_FLAT_CALL_COST_USD,
            latency_ms=latency_ms,
            raw={
                "model": self._model_name,
                "metric": "answer_relevancy",
                "backend": "ragas",
            },
        )