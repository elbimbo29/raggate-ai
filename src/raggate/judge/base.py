"""LLM judge interface.

A judge scores a generation result against a metric. Each call is
one question: "how faithful is this answer?", "how relevant is this
answer to the question?" — and returns a score in [0, 1].

Different backends (a fake for tests, DeepEval, RAGAS) implement the
same two methods. The harness doesn't care which one is underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class JudgeResult:
    """One judge call's outcome.

    score: 0.0 to 1.0, higher is better.
    reason: short explanation (may be empty for the fake judge).
    cost_usd: approximate cost of the call, 0.0 for fakes.
    latency_ms: wall-clock time for the call.
    raw: optional backend-specific payload for debugging.
    """

    score: float
    reason: str = ""
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class Judge(Protocol):
    """Scores a (question, answer, context, reference) tuple."""

    def faithfulness(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> JudgeResult:
        """Is the answer supported by the context? 1.0 = fully, 0.0 = not at all."""
        ...

    def answer_relevancy(self, question: str, answer: str) -> JudgeResult:
        """Does the answer address the question? 1.0 = fully, 0.0 = not at all."""
        ...

    def answer_correctness(
        self,
        question: str,
        answer: str,
        reference: str,
    ) -> JudgeResult:
        """Does the answer match the reference? 1.0 = matches, 0.0 = not at all."""
        ...

    @property
    def name(self) -> str:
        """Short identifier used in reports and dashboards."""
        ...