"""Tests for the API schemas.

Focus on validation rules: which inputs are accepted, which are
rejected, and what errors look like. These are the API's contract
with clients, so getting them right matters more than most tests.
"""

import pytest
from pydantic import ValidationError

from raggate.api.schemas import (
    CompareRequest,
    RunDetail,
    RunRequest,
    RunResponse,
    RunSummary,
)

# ---------- RunRequest ----------

def test_run_request_minimal_retrieval():
    req = RunRequest(kind="retrieval")
    assert req.kind == "retrieval"
    assert req.retriever == "chroma"
    assert req.k == 5
    assert req.generator is None
    assert req.judge is None


def test_run_request_generation_with_judge():
    req = RunRequest(kind="generation", generator="template", judge="deepeval")
    assert req.generator == "template"
    assert req.judge == "deepeval"


def test_run_request_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        RunRequest(kind="not_a_kind")


def test_run_request_rejects_k_zero():
    with pytest.raises(ValidationError):
        RunRequest(kind="retrieval", k=0)


def test_run_request_rejects_k_too_large():
    with pytest.raises(ValidationError):
        RunRequest(kind="retrieval", k=100)


# ---------- RunResponse ----------

def test_run_response_validates_status():
    resp = RunResponse(run_id="abc123", status="pending")
    assert resp.status == "pending"


def test_run_response_rejects_bad_status():
    with pytest.raises(ValidationError):
        RunResponse(run_id="abc123", status="done")  # not one of our literals


# ---------- RunSummary ----------

def test_run_summary_roundtrip():
    summary = RunSummary(
        id="abc123",
        kind="retrieval",
        status="succeeded",
        created_at="2026-10-05T00:00:00+00:00",
        retriever_name="chroma-minilm",
        generator_name=None,
        judge_name=None,
        k=5,
        n_cases=20,
        cost_usd=0.0,
        latency_ms=1234.5,
        metrics={"hit_rate@5": 0.9},
    )
    assert summary.metrics["hit_rate@5"] == 0.9

# ---------- RunDetail ----------

def test_run_detail_wraps_summary_and_cases():
    summary = RunSummary(
        id="abc123",
        kind="retrieval",
        status="succeeded",
        created_at="2026-10-05T00:00:00+00:00",
        retriever_name="chroma-minilm",
        generator_name=None,
        judge_name=None,
        k=5,
        n_cases=1,
        cost_usd=0.0,
        latency_ms=100.0,
        metrics={"mrr": 1.0},
    )
    detail = RunDetail(summary=summary, cases=[{"case_id": "auth-001"}])
    assert detail.cases[0]["case_id"] == "auth-001"

# ---------- CompareRequest ----------

def test_compare_request_minimal():
    req = CompareRequest(baseline_run_id="a", candidate_run_id="b")
    assert req.baseline_run_id == "a"


def test_compare_request_rejects_empty_ids():
    with pytest.raises(ValidationError):
        CompareRequest(baseline_run_id="", candidate_run_id="b")