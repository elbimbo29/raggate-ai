"""Tests for GET /eval/runs and GET /eval/runs/{id}."""

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


def _start_run(client, kind="retrieval", retriever="keyword", k=3):
    resp = client.post(
        "/eval/run",
        json={"kind": kind, "retriever": retriever, "k": k},
    )
    assert resp.status_code == 202
    return resp.json()["run_id"]


def test_list_runs_empty(client):
    resp = client.get("/eval/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_runs_after_a_run(client):
    run_id = _start_run(client)
    resp = client.get("/eval/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == run_id
    assert body[0]["status"] == "succeeded"
    assert body[0]["kind"] == "retrieval"


def test_list_runs_limit(client):
    for _ in range(3):
        _start_run(client)
    resp = client.get("/eval/runs?limit=2")
    assert len(resp.json()) == 2


def test_list_runs_invalid_limit(client):
    resp = client.get("/eval/runs?limit=0")
    assert resp.status_code == 422


def test_get_run_returns_detail_with_cases(client):
    run_id = _start_run(client)
    resp = client.get(f"/eval/runs/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["id"] == run_id
    assert body["summary"]["n_cases"] == 20
    assert len(body["cases"]) == 20
    # each case payload should have a case_id and retrieved_chunk_ids
    case = body["cases"][0]
    assert "case_id" in case
    assert "retrieved_chunk_ids" in case


def test_get_run_404_for_unknown_id(client):
    resp = client.get("/eval/runs/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]