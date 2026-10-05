"""Tests for POST /eval/run."""

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


def test_run_endpoint_accepts_retrieval(client):
    resp = client.post("/eval/run", json={"kind": "retrieval", "retriever": "keyword"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    assert len(body["run_id"]) == 32  # uuid4().hex


def test_run_endpoint_generation_requires_generator_and_judge(client):
    resp = client.post("/eval/run", json={"kind": "generation"})
    assert resp.status_code == 400
    assert "generator" in resp.json()["detail"]


def test_run_endpoint_rejects_bad_kind(client):
    resp = client.post("/eval/run", json={"kind": "potato"})
    assert resp.status_code == 422


def test_run_endpoint_rejects_bad_k(client):
    resp = client.post("/eval/run", json={"kind": "retrieval", "k": 0})
    assert resp.status_code == 422


def test_run_endpoint_persists_a_succeeded_run(client):
    """TestClient runs background tasks inline, so the run completes
    before this assertion."""
    resp = client.post(
        "/eval/run", json={"kind": "retrieval", "retriever": "keyword", "k": 3}
    )
    run_id = resp.json()["run_id"]

    from raggate.main import get_store
    rec = get_store().get_run(run_id)
    assert rec is not None
    assert rec.status == "succeeded"
    assert rec.n_cases == 20
    assert "hit_rate@3" in rec.metrics

    cases = get_store().get_run_cases(run_id)
    assert len(cases) == 20