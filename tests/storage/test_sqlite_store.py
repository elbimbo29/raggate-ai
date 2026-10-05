"""Tests for the SQLite run store.

Uses ':memory:' so nothing hits disk and tests are fast.
"""

from raggate.storage.base import CaseRecord, RunRecord
from raggate.storage.sqlite import SQLiteRunStore, new_run_id


def _store() -> SQLiteRunStore:
    return SQLiteRunStore(":memory:")


def _run(kind: str = "retrieval", **overrides) -> RunRecord:
    base = dict(
        id=new_run_id(),
        kind=kind,
        created_at="2026-10-05T00:00:00+00:00",
        retriever_name="chroma-minilm",
        generator_name=None,
        judge_name=None,
        k=5,
        n_cases=20,
        cost_usd=0.0,
        latency_ms=1234.5,
        metrics={"hit_rate@5": 0.9, "mrr": 0.85},
    )
    base.update(overrides)
    return RunRecord(**base)


def test_save_and_get_roundtrip():
    store = _store()
    rec = _run()
    store.save_run(rec, [CaseRecord(case_id="auth-001", payload={"score": 1.0})])

    fetched = store.get_run(rec.id)
    assert fetched is not None
    assert fetched.id == rec.id
    assert fetched.metrics == {"hit_rate@5": 0.9, "mrr": 0.85}


def test_get_missing_run_returns_none():
    store = _store()
    assert store.get_run("does-not-exist") is None


def test_get_run_cases_returns_all_ordered():
    store = _store()
    rec = _run()
    cases = [
        CaseRecord(case_id="b", payload={"x": 2}),
        CaseRecord(case_id="a", payload={"x": 1}),
    ]
    store.save_run(rec, cases)

    fetched = store.get_run_cases(rec.id)
    assert [c.case_id for c in fetched] == ["a", "b"]


def test_list_runs_filters_by_kind():
    store = _store()
    store.save_run(_run(kind="retrieval", created_at="2026-10-05T00:00:00+00:00"), [])
    store.save_run(_run(kind="generation", created_at="2026-10-05T01:00:00+00:00"), [])

    all_runs = store.list_runs()
    assert len(all_runs) == 2

    only_gen = store.list_runs(kind="generation")
    assert len(only_gen) == 1
    assert only_gen[0].kind == "generation"


def test_list_runs_newest_first():
    store = _store()
    old = _run(created_at="2026-10-05T00:00:00+00:00")
    new = _run(created_at="2026-10-05T02:00:00+00:00")
    store.save_run(old, [])
    store.save_run(new, [])

    listed = store.list_runs()
    assert listed[0].id == new.id
    assert listed[1].id == old.id


def test_save_run_is_atomic_on_duplicate_id():
    store = _store()
    rec = _run()
    store.save_run(rec, [CaseRecord(case_id="a", payload={})])

    # Saving the same run id again should fail, and the earlier cases
    # should remain untouched (no partial insert from the second call).
    import sqlite3

    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        store.save_run(rec, [CaseRecord(case_id="b", payload={})])

    cases = store.get_run_cases(rec.id)
    assert [c.case_id for c in cases] == ["a"]