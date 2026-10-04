"""Keyword baseline retriever.

Scores each chunk by how many distinct question tokens appear in the
chunk text. Deterministic, no dependencies, no model downloads. Used as
a baseline so the retrieval metrics can be tested in isolation, and as a
sanity floor — a real retriever should beat it.
"""

from __future__ import annotations

import re

from raggate.retriever.corpus import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class KeywordRetriever:
    """Retrieve chunks by token overlap with the question."""

    def __init__(self, chunks: list[Chunk], name: str = "keyword-v1") -> None:
        self._chunks = chunks
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, question: str, k: int = 5) -> list[str]:
        q_tokens = _tokenize(question)
        if not q_tokens:
            return []

        scored: list[tuple[int, str]] = []
        for chunk in self._chunks:
            overlap = len(q_tokens & _tokenize(chunk.text))
            if overlap > 0:
                scored.append((overlap, chunk.chunk_id))

        # Highest overlap first; tiebreak by chunk_id so output is stable.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [chunk_id for _, chunk_id in scored[:k]]