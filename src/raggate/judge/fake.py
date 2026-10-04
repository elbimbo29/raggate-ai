"""Fake judge.

Deterministic, dependency-free. Returns scores based on simple
heuristics so tests can assert expected values without hitting any
external service.

Heuristics:
  - faithfulness: fraction of answer words that appear in the context
  - answer_relevancy: fraction of question words that appear in the answer
  - answer_correctness: word-overlap between answer and reference

These are NOT good metrics — they're stand-ins so tests stay fast and
free. Real metrics use the DeepEval judge (Step 3.5).
"""

from __future__ import annotations

import re

from raggate.judge.base import JudgeResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class FakeJudge:
    """Deterministic judge for tests."""

    def __init__(self, name: str = "fake-judge") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def faithfulness(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> JudgeResult:
        answer_tokens = _tokens(answer)
        if not answer_tokens:
            return JudgeResult(score=0.0, reason="empty answer")
        context_tokens = _tokens(" ".join(context))
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        return JudgeResult(score=overlap, reason="word overlap with context")

    def answer_relevancy(self, question: str, answer: str) -> JudgeResult:
        q_tokens = _tokens(question)
        if not q_tokens:
            return JudgeResult(score=0.0, reason="empty question")
        a_tokens = _tokens(answer)
        overlap = len(q_tokens & a_tokens) / len(q_tokens)
        return JudgeResult(score=overlap, reason="word overlap with question")

    def answer_correctness(
        self,
        question: str,
        answer: str,
        reference: str,
    ) -> JudgeResult:
        ref_tokens = _tokens(reference)
        if not ref_tokens:
            return JudgeResult(score=0.0, reason="empty reference")
        a_tokens = _tokens(answer)
        overlap = len(ref_tokens & a_tokens) / len(ref_tokens)
        return JudgeResult(score=overlap, reason="word overlap with reference")