"""CLI for the regression gate.

Usage:
    raggate-gate check --baseline <run_id> --candidate <run_id>
    raggate-gate check --baseline <id> --candidate <id> --kind generation
    raggate-gate check --baseline <id> --candidate <id> --json

Exit codes:
    0 — gate passed
    1 — gate failed (regression detected)
    2 — error (config, missing run, kind mismatch, etc.)

The exit code is the whole point: CI uses it to decide whether to block.
"""

from __future__ import annotations

import argparse
import json
import sys

from raggate.gate.config import (
    DEFAULT_PATH as DEFAULT_THRESHOLDS_PATH,
)
from raggate.gate.config import ThresholdsConfigError, load_thresholds
from raggate.gate.evaluator import evaluate_gate
from raggate.storage.sqlite import SQLiteRunStore


def _cmd_check(args: argparse.Namespace) -> int:
    # Load store
    store = SQLiteRunStore(args.db) if args.db else SQLiteRunStore()

    baseline = store.get_run(args.baseline)
    if baseline is None:
        print(f"[ERROR] baseline run not found: {args.baseline}", file=sys.stderr)
        return 2

    candidate = store.get_run(args.candidate)
    if candidate is None:
        print(
            f"[ERROR] candidate run not found: {args.candidate}", file=sys.stderr
        )
        return 2

    if baseline.kind != candidate.kind:
        print(
            f"[ERROR] cannot gate different kinds: "
            f"{baseline.kind} vs {candidate.kind}",
            file=sys.stderr,
        )
        return 2

    if baseline.status != "succeeded":
        print(
            f"[ERROR] baseline run is not finished: status={baseline.status!r}",
            file=sys.stderr,
        )
        return 2
    if candidate.status != "succeeded":
        print(
            f"[ERROR] candidate run is not finished: status={candidate.status!r}",
            file=sys.stderr,
        )
        return 2

    # Load thresholds (CLI arg overrides default path)
    kind = args.kind or baseline.kind
    try:
        thresholds = load_thresholds(kind, args.thresholds)
    except ThresholdsConfigError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    result = evaluate_gate(
        baseline_metrics=baseline.metrics,
        candidate_metrics=candidate.metrics,
        thresholds=thresholds,
    )

    if args.json:
        payload = {
            "passed": result.passed,
            "kind": kind,
            "baseline_run_id": baseline.id,
            "candidate_run_id": candidate.id,
            "baseline_metrics": baseline.metrics,
            "candidate_metrics": candidate.metrics,
            "regressions": [r.model_dump() for r in result.regressions],
            "improvements": [r.model_dump() for r in result.improvements],
            "unchanged": result.unchanged,
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_human(result, baseline.id, candidate.id, kind)

    return 0 if result.passed else 1


def _print_human(result, baseline_id: str, candidate_id: str, kind: str) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] gate  kind={kind}")
    print(f"  baseline:  {baseline_id}")
    print(f"  candidate: {candidate_id}")
    print()

    if result.regressions:
        print("  regressions:")
        for r in result.regressions:
            marker = "*" if r.threshold > 0 else " "
            print(
                f"   {marker} {r.metric:<22} "
                f"baseline={r.baseline:.4f}  "
                f"candidate={r.candidate:.4f}  "
                f"delta={r.delta:+.4f}  "
                f"threshold={r.threshold:.4f}"
            )
        print("    (* = gated metric that failed)")

    if result.improvements:
        print("  improvements:")
        for i in result.improvements:
            print(
                f"    {i.metric:<22} "
                f"baseline={i.baseline:.4f}  "
                f"candidate={i.candidate:.4f}  "
                f"delta={i.delta:+.4f}"
            )

    if result.unchanged:
        print(f"  unchanged: {', '.join(result.unchanged)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raggate-gate",
        description="Run the RAGGate AI regression gate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser(
        "check", help="Compare two runs and exit non-zero on regression."
    )
    p_check.add_argument("--baseline", required=True, help="Baseline run ID.")
    p_check.add_argument("--candidate", required=True, help="Candidate run ID.")
    p_check.add_argument(
        "--kind",
        default=None,
        choices=["retrieval", "generation"],
        help="Override run kind (default: from baseline record).",
    )
    p_check.add_argument(
        "--thresholds",
        default=str(DEFAULT_THRESHOLDS_PATH),
        help=f"Path to thresholds file (default: {DEFAULT_THRESHOLDS_PATH}).",
    )
    p_check.add_argument(
        "--db",
        default=None,
        help="Path to SQLite DB. Defaults to the app's configured DB.",
    )
    p_check.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    p_check.set_defaults(func=_cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())