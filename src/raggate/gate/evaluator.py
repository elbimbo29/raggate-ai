"""Pure-function gate evaluator.

Takes two metric dicts and a threshold set and produces a GateResult.
No I/O, no HTTP, no storage. Everything else in the gate package
(API endpoint, CLI) wraps this function.

Only metrics with an explicit threshold can fail the gate. Metrics
without a threshold are still reported (as regressions or improvements)
so the report is informative, but they don't affect the pass/fail
verdict. This is deliberate: adding a new metric to the harness
shouldn't silently start failing builds.
"""

from __future__ import annotations

from raggate.gate.models import GateResult, MetricDelta, Thresholds


def evaluate_gate(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    thresholds: Thresholds,
) -> GateResult:
    """Compare candidate vs baseline against thresholds.

    A gated metric fails if (baseline - candidate) > threshold. Metrics
    without a threshold are reported but never fail the gate. Metrics
    in thresholds but missing from candidate are treated as a drop to
    0.0, which will usually fail.
    """
    regressions: list[MetricDelta] = []
    improvements: list[MetricDelta] = []
    unchanged: list[str] = []
    gated_regressions: list[MetricDelta] = []

    all_metrics = sorted(
        set(thresholds.per_metric)
        | set(baseline_metrics)
        | set(candidate_metrics)
    )

    for name in all_metrics:
        baseline = baseline_metrics.get(name, 0.0)
        candidate = candidate_metrics.get(name, 0.0)
        delta = candidate - baseline
        threshold = thresholds.per_metric.get(name)
        is_gated = threshold is not None

        if delta > 0:
            # Improvement, gated or not.
            improvements.append(
                MetricDelta(
                    metric=name,
                    baseline=baseline,
                    candidate=candidate,
                    delta=delta,
                    threshold=threshold if is_gated else 0.0,
                )
            )
        elif is_gated and (baseline - candidate) > threshold:
            # Gated metric dropped past its threshold -> fails the gate.
            entry = MetricDelta(
                metric=name,
                baseline=baseline,
                candidate=candidate,
                delta=delta,
                threshold=threshold,
            )
            regressions.append(entry)
            gated_regressions.append(entry)
        elif is_gated:
            # Gated metric dropped, but within the allowed tolerance.
            unchanged.append(name)
        elif delta < 0:
            # Non-gated drop. Reported but doesn't fail the gate.
            regressions.append(
                MetricDelta(
                    metric=name,
                    baseline=baseline,
                    candidate=candidate,
                    delta=delta,
                    threshold=0.0,
                )
            )
        else:
            # No movement.
            unchanged.append(name)

    regressions.sort(key=lambda r: r.delta)
    improvements.sort(key=lambda r: r.delta, reverse=True)

    return GateResult(
        passed=len(gated_regressions) == 0,
        regressions=regressions,
        improvements=improvements,
        unchanged=unchanged,
    )