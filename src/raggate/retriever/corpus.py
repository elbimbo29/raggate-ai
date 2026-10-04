"""Load the AcmeDB corpus from JSONL into typed Chunk objects.

The corpus is the set of documents the retriever searches over. Each
chunk is one small unit of text with a stable chunk_id that the golden
set references via expected_chunk_ids.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class CorpusError(Exception):
    """Raised when the corpus file fails to load or validate."""


class Chunk(BaseModel):
    chunk_id: str = Field(..., min_length=1)
    section: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


def load_corpus(path: str | Path) -> list[Chunk]:
    """Read a JSONL corpus and return validated Chunk objects.

    Fails with CorpusError (line-numbered) if the file is missing,
    a line is invalid JSON, a line fails Chunk validation, or
    duplicate chunk_ids are found.
    """
    path = Path(path)
    if not path.exists():
        raise CorpusError(f"Corpus not found: {path}")

    chunks: list[Chunk] = []
    errors: list[str] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  line {lineno}: invalid JSON ({e.msg})")
                continue

            try:
                chunk = Chunk.model_validate(payload)
            except Exception as e:  # pydantic ValidationError
                errors.append(f"  line {lineno}: {e}")
                continue

            if chunk.chunk_id in seen:
                errors.append(f"  line {lineno}: duplicate chunk_id '{chunk.chunk_id}'")
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)

    if errors:
        raise CorpusError(f"Failed to load corpus from {path}:\n" + "\n".join(errors))

    if not chunks:
        raise CorpusError(f"Corpus is empty: {path}")

    return chunks