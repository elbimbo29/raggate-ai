"""Tests for POST /eval/gate."""

import pytest
from fastapi.testclient import TestClient

from raggate.main import app, set_store
from raggate.storage.sqlite import SQLiteRunStore


@pytest.fixture
def client():
    store = SQLiteRunStore(":memory:")
    set_store(store)
    with TestClient(app) as c:
        yield c
    store.close()
    set_store(None)


def _start_run(client, retriever="keyword", k=3):
    resp = client.post(
        "/eval/run",
        json={"kind": "retrieval", "retriever": retriever, "k": k},
    )
    assert resp.status_code == 202
    return resp.json()["run_id"]


def test_gate_passes_with_generous_thresholds(client):
    # Same config -> identical metrics -> passes.
    base_id = _start_run(client, retriever="keyword")
    cand_id = _start_run(client, retriever="keyword")

    resp = client.post(
        "/eval/gate",
        json={
            "baseline_run_id": base_id,
            "candidate_run_id": cand_id,
            "thresholds": {"hit_rate@3": 0.01, "mrr": 0.01},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert body["regressions"] == []


def test_gate_fails_when_retriever_regresses(client):
    # Baseline = good (chroma). Candidate = bad (keyword).
    # At k=3 the drop is ~0.05 on hit_rate and ~0.04 on mrr, so a
    # 0.05 threshold sits right on the boundary and passes. We use
    # 0.02 to catch the regression — a realistic tightening.
    base_id = _start_run(client, retriever="chroma")
    cand_id = _start_run(client, retriever="keyword")

    resp = client.post(
        "/eval/gate",
        json={
            "baseline_run_id": base_id,
            "candidate_run_id": cand_id,
            "thresholds": {"hit_rate@3": 0.02, "mrr": 0.02},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is False
    names = [r["metric"] for r in body["regressions"]]
    assert "hit_rate@3" in names
    assert "mrr" in names


def test_gate_missing_baseline_404(client):
    cand_id = _start_run(client)
    resp = client.post(
        "/eval/gate",
        json={
            "baseline_run_id": "does-not-exist",
            "candidate_run_id": cand_id,
            "thresholds": {"mrr": 0.05},
        },
    )
    assert resp.status_code == 404
    assert "baseline" in resp.json()["detail"]


def test_gate_kind_mismatch_400(client):
    retrieval_id = _start_run(client)

    from datetime import UTC, datetime

    from raggate.main import get_store
    from raggate.storage.base import RunRecord

    gen_id = "fake-gen-run"
    gen = RunRecord(
        id=gen_id,
        kind="generation",
        created_at=datetime.now(UTC).isoformat(),
        retriever_name="chroma-minilm",
        generator_name="template-v1",
        judge_name="deepeval-gpt-4o-mini",
        k=3,
        n_cases=1,
        cost_usd=0.0,
        latency_ms=0.0,
        metrics={"faithfulness": 0.9},
        status="succeeded",
        error=None,
    )
    get_store().save_run(gen, [])

    resp = client.post(
        "/eval/gate",
        json={
            "baseline_run_id": retrieval_id,
            "candidate_run_id": gen_id,
            "thresholds": {"mrr": 0.05},
        },
    )
    assert resp.status_code == 400
    assert "different run kinds" in resp.json()["detail"]