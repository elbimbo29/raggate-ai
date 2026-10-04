"""Retriever interface.

A retriever takes a question and returns the top-k chunk IDs from the
corpus, ranked best-first. Different backends (keyword baseline, Chroma,
a real RAG pipeline) implement this same method so the harness doesn't
care which one it's scoring.
"""

from __future__ import annotations

from typing import Protocol


class Retriever(Protocol):
    """Ranked-chunk retriever interface."""

    def retrieve(self, question: str, k: int = 5) -> list[str]:
        """Return the top-k chunk IDs for the question, best-first."""
        ...

    @property
    def name(self) -> str:
        """Short identifier used in reports and dashboards."""
        ...