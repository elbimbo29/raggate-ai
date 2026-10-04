"""Generator interface.

A generator takes a question plus the retrieved context chunks and
returns an answer. Different backends (a template stitcher, an
OpenAI-backed RAG, a real pipeline) implement this same method so the
harness doesn't care which one it's scoring.
"""

from __future__ import annotations

from typing import Protocol

from raggate.retriever.corpus import Chunk


class Generator(Protocol):
    """Question + context -> answer."""

    def generate(self, question: str, context: list[Chunk]) -> str:
        """Produce an answer using only the provided context."""
        ...

    @property
    def name(self) -> str:
        """Short identifier used in reports and dashboards."""
        ...