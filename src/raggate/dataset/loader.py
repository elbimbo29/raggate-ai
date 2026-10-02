"""Load and validate the golden set from a JSONL file.

Each line is a JSON object matching the EvalCase schema. The loader
fails loudly and precisely: on a validation error it raises
DatasetError with the offending line number(s), so a broken golden
set is easy to fix rather than a wall of traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from raggate.dataset.models import EvalCase


class DatasetError(Exception):
    """Raised when the golden set fails to load or validate."""


def load_cases(path: str | Path) -> list[EvalCase]:
    """Read a JSONL golden set and return validated EvalCase objects.

    Raises DatasetError if the file is missing, a line is not valid
    JSON, or any line fails EvalCase validation. Errors are collected
    across all lines so one bad row doesn't hide the rest.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Golden set not found: {path}")

    cases: list[EvalCase] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  line {lineno}: invalid JSON ({e.msg})")
                continue

            try:
                case = EvalCase.model_validate(payload)
            except ValidationError as e:
                first = e.errors()[0]
                loc = ".".join(str(x) for x in first["loc"])
                errors.append(f"  line {lineno}: {loc}: {first['msg']}")
                continue

            if case.id in seen_ids:
                errors.append(f"  line {lineno}: duplicate id '{case.id}'")
                continue
            seen_ids.add(case.id)
            cases.append(case)

    if errors:
        raise DatasetError(
            f"Failed to load golden set from {path}:\n" + "\n".join(errors)
        )

    if not cases:
        raise DatasetError(f"Golden set is empty: {path}")

    return cases