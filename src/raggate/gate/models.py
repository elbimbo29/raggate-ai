"""Models for the regression gate.

The gate compares a baseline run's metrics to a candidate run's metrics
and decides pass/fail against a set of thresholds.

A threshold is a positive number meaning "max allowed drop." So
{"hit_rate@5": 0.05} means the candidate may be at most 0.05 lower
than the baseline before the gate fails.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Thresholds(BaseModel):
    """Per-metric max allowed drop, as absolute values."""

    per_metric: dict[str, float] = Field(
        ...,
        description=(
            "metric name -> max allowed absolute drop. "
            "A positive value. Example: {'mrr': 0.05}."
        ),
    )


class MetricDelta(BaseModel):
    """One metric's baseline/candidate/delta values."""

    metric: str
    baseline: float
    candidate: float
    delta: float = Field(..., description="candidate - baseline")
    threshold: float = Field(..., description="max allowed drop (positive)")


class GateResult(BaseModel):
    """Outcome of a gate evaluation."""

    passed: bool
    regressions: list[MetricDelta]
    improvements: list[MetricDelta]
    unchanged: list[str]