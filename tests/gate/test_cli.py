"""Tests for the gate CLI.

Tests use subprocess-free invocation (call main() directly) plus a
real SQLite store so exit codes and output are exercised end-to-end
without spawning a process.
"""

import json

from raggate.gate.cli import main
from raggate.storage.base import RunRecord
from raggate.storage.sqlite import SQLiteRunStore


def _store(tmp_path) -> tuple[SQLiteRunStore, str, str]:
    """Create a store with a good and a bad run. Returns (store, good_id, bad_id)."""
    db = tmp_path / "test.sqlite"
    store = SQLiteRunStore(str(db))

    good = RunRecord(
        id="good-run",
        kind="retrieval",
        created_at="2026-10-05T00:00:00+00:00",
        retriever_name="chroma-minilm",
        generator_name=None,
        judge_name=None,
        k=5,
        n_cases=20,
        cost_usd=0.0,
        latency_ms=0.0,
        metrics={"hit_rate@5": 1.0, "mrr": 0.9},
        status="succeeded",
        error=None,
    )
    bad = RunRecord(
        id="bad-run",
        kind="retrieval",
        created_at="2026-10-05T01:00:00+00:00",
        retriever_name="keyword-v1",
        generator_name=None,
        judge_name=None,
        k=5,
        n_cases=20,
        cost_usd=0.0,
        latency_ms=0.0,
        metrics={"hit_rate@5": 0.7, "mrr": 0.6},
        status="succeeded",
        error=None,
    )
    store.save_run(good, [])
    store.save_run(bad, [])
    return store, "good-run", "bad-run"


def _thresholds_file(tmp_path) -> str:
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"retrieval": {"hit_rate@5": 0.02, "mrr": 0.02}}),
        encoding="utf-8",
    )
    return str(path)


def test_check_passes_when_candidate_matches(tmp_path, capsys):
    store, good_id, _ = _store(tmp_path)
    code = main(
        [
            "check",
            "--baseline",
            good_id,
            "--candidate",
            good_id,
            "--thresholds",
            _thresholds_file(tmp_path),
            "--db",
            str(tmp_path / "test.sqlite"),
        ]
    )
    assert code == 0
    assert "PASS" in capsys.readouterr().out


def test_check_fails_on_regression(tmp_path, capsys):
    store, good_id, bad_id = _store(tmp_path)
    code = main(
        [
            "check",
            "--baseline",
            good_id,
            "--candidate",
            bad_id,
            "--thresholds",
            _thresholds_file(tmp_path),
            "--db",
            str(tmp_path / "test.sqlite"),
        ]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "hit_rate@5" in out


def test_check_errors_on_missing_run(tmp_path):
    store, _, _ = _store(tmp_path)
    code = main(
        [
            "check",
            "--baseline",
            "does-not-exist",
            "--candidate",
            "good-run",
            "--thresholds",
            _thresholds_file(tmp_path),
            "--db",
            str(tmp_path / "test.sqlite"),
        ]
    )
    assert code == 2


def test_check_json_output(tmp_path, capsys):
    store, good_id, bad_id = _store(tmp_path)
    code = main(
        [
            "check",
            "--baseline",
            good_id,
            "--candidate",
            bad_id,
            "--thresholds",
            _thresholds_file(tmp_path),
            "--db",
            str(tmp_path / "test.sqlite"),
            "--json",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert any(r["metric"] == "hit_rate@5" for r in payload["regressions"])