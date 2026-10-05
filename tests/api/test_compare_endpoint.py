"""Tests for POST /eval/compare."""

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


def test_compare_two_retrieval_runs(client):
    baseline_id = _start_run(client)
    candidate_id = _start_run(client)

    resp = client.post(
        "/eval/compare",
        json={
            "baseline_run_id": baseline_id,
            "candidate_run_id": candidate_id,
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["baseline_run_id"] == baseline_id
    assert body["candidate_run_id"] == candidate_id
    assert body["kind"] == "retrieval"

    # Same inputs -> deltas should be exactly zero on every metric.
    assert set(body["deltas"].keys()) == set(body["baseline_metrics"].keys())
    for name, delta in body["deltas"].items():
        assert delta == pytest.approx(0.0, abs=1e-9), f"expected 0 delta for {name}"


def test_compare_missing_baseline_404(client):
    candidate_id = _start_run(client)
    resp = client.post(
        "/eval/compare",
        json={
            "baseline_run_id": "does-not-exist",
            "candidate_run_id": candidate_id,
        },
    )
    assert resp.status_code == 404
    assert "baseline" in resp.json()["detail"]


def test_compare_missing_candidate_404(client):
    baseline_id = _start_run(client)
    resp = client.post(
        "/eval/compare",
        json={
            "baseline_run_id": baseline_id,
            "candidate_run_id": "does-not-exist",
        },
    )
    assert resp.status_code == 404
    assert "candidate" in resp.json()["detail"]


def test_compare_same_run_returns_zero_deltas(client):
    run_id = _start_run(client)
    resp = client.post(
        "/eval/compare",
        json={"baseline_run_id": run_id, "candidate_run_id": run_id},
    )
    assert resp.status_code == 200
    for delta in resp.json()["deltas"].values():
        assert delta == pytest.approx(0.0)


def test_compare_kind_mismatch_400(client):
    # retrieval run
    retrieval_id = _start_run(client, k=3)

    # Fake a generation run directly into the store — bypass the
    # endpoint so we don't have to pay for a real LLM judge call.
    from datetime import UTC, datetime

    from raggate.main import get_store
    from raggate.storage.base import RunRecord

    gen_id = "fake-gen-run"
    gen_record = RunRecord(
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
    get_store().save_run(gen_record, [])

    resp = client.post(
        "/eval/compare",
        json={"baseline_run_id": retrieval_id, "candidate_run_id": gen_id},
    )
    assert resp.status_code == 400
    assert "different run kinds" in resp.json()["detail"]