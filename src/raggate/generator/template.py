"""Template generator.

Concatenates the retrieved context chunks into a single string,
trimmed to a reasonable length. No LLM involved. Deterministic, fast,
free. Used for testing the generation harness end-to-end without an
API key, and as a floor in reports — a real generator should beat it
on answer relevancy (the template ignores the question entirely) but
it will score high on faithfulness (it copies from context).
"""

from __future__ import annotations

from raggate.retriever.corpus import Chunk

MAX_CHARS = 600


class TemplateGenerator:
    """Answer = joined top context chunks, truncated."""

    def __init__(self, name: str = "template-v1") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(self, question: str, context: list[Chunk]) -> str:
        # The question is deliberately ignored. The template generator
        # is meant to be weak on relevancy — a negative control for the
        # generation metrics, just like KeywordRetriever is for retrieval.
        if not context:
            return ""
        joined = " ".join(c.text.strip() for c in context)
        if len(joined) > MAX_CHARS:
            joined = joined[:MAX_CHARS].rsplit(" ", 1)[0] + "..."
        return joined