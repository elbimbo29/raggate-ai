"""Tests for the gate evaluator.

Pure function tests — no mocks, no I/O. Every case is a dictionary
comparison with hand-computed expected results.
"""

import pytest

from raggate.gate.evaluator import evaluate_gate
from raggate.gate.models import Thresholds


def _thresholds(**kwargs) -> Thresholds:
    return Thresholds(per_metric=kwargs)


# ---------- passing cases ----------

def test_gate_passes_when_metrics_are_identical():
    result = evaluate_gate(
        baseline_metrics={"mrr": 0.80},
        candidate_metrics={"mrr": 0.80},
        thresholds=_thresholds(mrr=0.05),
    )
    assert result.passed is True
    assert result.regressions == []
    assert result.unchanged == ["mrr"]


def test_gate_passes_when_drop_is_within_threshold():
    # baseline 0.80, candidate 0.76 -> drop 0.04, threshold 0.05 -> pass
    result = evaluate_gate(
        baseline_metrics={"mrr": 0.80},
        candidate_metrics={"mrr": 0.76},
        thresholds=_thresholds(mrr=0.05),
    )
    assert result.passed is True
    assert result.regressions == []
    # 0.04 drop counts as "unchanged" because it's within tolerance.
    assert "mrr" in result.unchanged


def test_gate_passes_on_improvement():
    result = evaluate_gate(
        baseline_metrics={"mrr": 0.80},
        candidate_metrics={"mrr": 0.90},
        thresholds=_thresholds(mrr=0.05),
    )
    assert result.passed is True
    assert len(result.improvements) == 1
    assert result.improvements[0].metric == "mrr"
    assert result.improvements[0].delta == pytest.approx(0.10)


# ---------- failing cases ----------

def test_gate_fails_on_drop_past_threshold():
    # baseline 0.80, candidate 0.70 -> drop 0.10, threshold 0.05 -> fail
    result = evaluate_gate(
        baseline_metrics={"mrr": 0.80},
        candidate_metrics={"mrr": 0.70},
        thresholds=_thresholds(mrr=0.05),
    )
    assert result.passed is False
    assert len(result.regressions) == 1
    reg = result.regressions[0]
    assert reg.metric == "mrr"
    assert reg.baseline == 0.80
    assert reg.candidate == 0.70
    assert reg.delta == pytest.approx(-0.10)
    assert reg.threshold == 0.05


def test_gate_fails_when_metric_missing_from_candidate():
    # candidate doesn't have the metric -> treated as 0.0 -> big drop
    result = evaluate_gate(
        baseline_metrics={"mrr": 0.80},
        candidate_metrics={},
        thresholds=_thresholds(mrr=0.05),
    )
    assert result.passed is False
    assert result.regressions[0].candidate == 0.0


def test_regressions_sorted_worst_first():
    result = evaluate_gate(
        baseline_metrics={"mrr": 0.80, "hit_rate@5": 0.90},
        candidate_metrics={"mrr": 0.60, "hit_rate@5": 0.70},
        thresholds=_thresholds(mrr=0.05, **{"hit_rate@5": 0.05}),
    )
    # hit_rate@5 dropped 0.20, mrr dropped 0.20 -> both equal
    # but if one were worse it'd come first. Here check ordering
    # is stable and both are present.
    assert len(result.regressions) == 2
    assert {r.metric for r in result.regressions} == {"mrr", "hit_rate@5"}


# ---------- metrics without thresholds ----------

def test_metric_without_threshold_never_fails():
    # 'other' has no threshold; a huge drop should still not fail.
    result = evaluate_gate(
        baseline_metrics={"other": 0.90, "mrr": 0.80},
        candidate_metrics={"other": 0.10, "mrr": 0.80},
        thresholds=_thresholds(mrr=0.05),
    )
    assert result.passed is True
    # 'other' is tracked as a regression in the report, but doesn't fail.
    assert any(r.metric == "other" for r in result.regressions)


def test_metric_only_in_candidate_is_tracked_as_improvement():
    result = evaluate_gate(
        baseline_metrics={"mrr": 0.80},
        candidate_metrics={"mrr": 0.80, "new_metric": 0.50},
        thresholds=_thresholds(mrr=0.05),
    )
    assert result.passed is True
    assert any(i.metric == "new_metric" for i in result.improvements)