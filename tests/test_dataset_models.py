"""Tests for the EvalCase / RetrievalResult / GenerationResult models."""

import pytest
from pydantic import ValidationError

from raggate.dataset.models import (
    Category,
    Difficulty,
    EvalCase,
    GenerationResult,
    RetrievalResult,
)


def _valid_case(**overrides) -> dict:
    base = {
        "id": "auth-001",
        "question": "How do I authenticate?",
        "expected_answer": "Use a Bearer token.",
        "expected_chunk_ids": ["chunk-auth-01"],
        "category": "auth",
        "difficulty": "easy",
    }
    base.update(overrides)
    return base


def test_valid_case_parses():
    case = EvalCase.model_validate(_valid_case())
    assert case.id == "auth-001"
    assert case.category is Category.AUTH
    assert case.difficulty is Difficulty.EASY
    assert case.notes is None


def test_id_rejects_uppercase():
    with pytest.raises(ValidationError):
        EvalCase.model_validate(_valid_case(id="Auth-001"))


def test_id_rejects_underscore():
    with pytest.raises(ValidationError):
        EvalCase.model_validate(_valid_case(id="auth_001"))


def test_expected_chunk_ids_must_be_nonempty():
    with pytest.raises(ValidationError):
        EvalCase.model_validate(_valid_case(expected_chunk_ids=[]))


def test_expected_chunk_ids_reject_duplicates():
    with pytest.raises(ValidationError):
        EvalCase.model_validate(
            _valid_case(expected_chunk_ids=["chunk-auth-01", "chunk-auth-01"])
        )


def test_invalid_category_rejected():
    with pytest.raises(ValidationError):
        EvalCase.model_validate(_valid_case(category="not_a_category"))


def test_retrieval_scores_length_must_match_chunks():
    with pytest.raises(ValidationError):
        RetrievalResult(
            case_id="auth-001",
            retrieved_chunk_ids=["c1", "c2"],
            scores=[0.9],
        )


def test_retrieval_scores_optional():
    r = RetrievalResult(case_id="auth-001", retrieved_chunk_ids=["c1", "c2"])
    assert r.scores is None


def test_generation_result_minimal():
    g = GenerationResult(
        case_id="auth-001",
        answer="Use a Bearer token.",
        retrieved_chunk_ids=["chunk-auth-01"],
    )
    assert g.latency_ms is None