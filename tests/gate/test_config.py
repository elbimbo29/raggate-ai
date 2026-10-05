"""Tests for the thresholds config loader."""

import json
from pathlib import Path

import pytest

from raggate.gate.config import ThresholdsConfigError, load_thresholds


def _write(tmp_path, payload):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_retrieval_thresholds(tmp_path):
    path = _write(
        tmp_path,
        {
            "retrieval": {"hit_rate@5": 0.02, "mrr": 0.02},
            "generation": {"faithfulness": 0.10},
        },
    )
    thresholds = load_thresholds("retrieval", path)
    assert thresholds.per_metric == {"hit_rate@5": 0.02, "mrr": 0.02}


def test_loads_generation_thresholds(tmp_path):
    path = _write(
        tmp_path,
        {
            "retrieval": {"mrr": 0.02},
            "generation": {"faithfulness": 0.10, "answer_relevancy": 0.15},
        },
    )
    thresholds = load_thresholds("generation", path)
    assert thresholds.per_metric == {
        "faithfulness": 0.10,
        "answer_relevancy": 0.15,
    }


def test_missing_file_raises(tmp_path):
    with pytest.raises(ThresholdsConfigError, match="not found"):
        load_thresholds("retrieval", tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ThresholdsConfigError, match="invalid JSON"):
        load_thresholds("retrieval", path)


def test_missing_kind_raises(tmp_path):
    path = _write(tmp_path, {"retrieval": {"mrr": 0.02}})
    with pytest.raises(
        ThresholdsConfigError, match="no thresholds for kind 'generation'"
    ):
        load_thresholds("generation", path)


def test_empty_kind_block_raises(tmp_path):
    path = _write(tmp_path, {"retrieval": {}})
    with pytest.raises(ThresholdsConfigError, match="non-empty object"):
        load_thresholds("retrieval", path)


def test_negative_threshold_raises(tmp_path):
    path = _write(tmp_path, {"retrieval": {"mrr": -0.05}})
    with pytest.raises(ThresholdsConfigError, match="non-negative"):
        load_thresholds("retrieval", path)


def test_non_numeric_threshold_raises(tmp_path):
    path = _write(tmp_path, {"retrieval": {"mrr": "loose"}})
    with pytest.raises(ThresholdsConfigError, match="non-negative"):
        load_thresholds("retrieval", path)


def test_shipped_thresholds_file_is_valid():
    """The thresholds file committed to the repo must always load."""
    repo_root = Path(__file__).resolve().parents[2]
    retrieval = load_thresholds(
        "retrieval", repo_root / "config" / "thresholds.json"
    )
    generation = load_thresholds(
        "generation", repo_root / "config" / "thresholds.json"
    )
    assert "mrr" in retrieval.per_metric
    assert "faithfulness" in generation.per_metric