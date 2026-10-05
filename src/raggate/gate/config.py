"""Load gate thresholds from a config file.

Thresholds live in a JSON file so they're version-controlled and
reviewable alongside the code. A PR that loosens a threshold is a PR
that changes what the gate catches — it should be visible in review.

The file is grouped by run kind ('retrieval', 'generation') because
each kind has its own metric set.

Retrieval thresholds are typically tight (deterministic metrics, no
noise). Generation thresholds are looser because LLM-judged metrics
vary run to run even on identical inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from raggate.gate.models import Thresholds

DEFAULT_PATH = Path("config/thresholds.json")


class ThresholdsConfigError(Exception):
    """Raised when the thresholds file can't be loaded or validated."""


def load_thresholds(
    kind: str,
    path: str | Path = DEFAULT_PATH,
) -> Thresholds:
    """Load thresholds for the given run kind.

    kind: 'retrieval' or 'generation'.
    path: JSON file with top-level keys matching run kinds.
    """
    path = Path(path)
    if not path.exists():
        raise ThresholdsConfigError(f"thresholds file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ThresholdsConfigError(
            f"invalid JSON in {path}: {e.msg} (line {e.lineno})"
        ) from e

    if not isinstance(payload, dict):
        raise ThresholdsConfigError(
            f"{path}: top-level must be an object keyed by run kind"
        )

    if kind not in payload:
        raise ThresholdsConfigError(
            f"{path}: no thresholds for kind {kind!r}. "
            f"Available kinds: {sorted(payload.keys())}"
        )

    per_metric = payload[kind]
    if not isinstance(per_metric, dict) or not per_metric:
        raise ThresholdsConfigError(
            f"{path}: thresholds for {kind!r} must be a non-empty object"
        )

    for name, value in per_metric.items():
        if not isinstance(value, (int, float)) or value < 0:
            raise ThresholdsConfigError(
                f"{path}: threshold for {name!r} must be a non-negative number, "
                f"got {value!r}"
            )

    return Thresholds(per_metric=per_metric)