"""Pydantic models for the golden dataset.

An EvalCase is one row of the golden set: a question, its ground-truth
answer, and the corpus chunks that should be retrieved to answer it.
RetrievalResult and GenerationResult are what the harness produces when
it runs a pipeline against a case.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Category(StrEnum):
    ...
    AUTH = "auth"
    ENDPOINTS = "endpoints"
    DATA_MODEL = "data_model"
    LIMITS = "limits"
    ERRORS = "errors"
    MISC = "misc"


class Difficulty(StrEnum):
    ...
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvalCase(BaseModel):
    """One golden-set case: question + expected answer + expected sources."""

    id: str = Field(..., min_length=1, pattern=r"^[a-z0-9-]+$")
    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    expected_chunk_ids: list[str] = Field(..., min_length=1)
    category: Category
    difficulty: Difficulty
    notes: str | None = None

    @field_validator("expected_chunk_ids")
    @classmethod
    def no_duplicate_chunks(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("expected_chunk_ids contains duplicates")
        return v


class RetrievalResult(BaseModel):
    """What the retriever returned for one case, ranked best-first."""

    case_id: str
    retrieved_chunk_ids: list[str]
    scores: list[float] | None = None

    @field_validator("scores")
    @classmethod
    def scores_match_chunks(cls, v: list[float] | None, info) -> list[float] | None:
        if v is None:
            return v
        chunks = info.data.get("retrieved_chunk_ids")
        if chunks is not None and len(v) != len(chunks):
            raise ValueError("scores length must match retrieved_chunk_ids length")
        return v


class GenerationResult(BaseModel):
    """What the LLM answered for one case."""

    case_id: str
    answer: str
    retrieved_chunk_ids: list[str]
    latency_ms: float | None = None