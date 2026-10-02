"""Command-line interface for the golden dataset.

Usage:
    raggate-dataset validate [path]

Defaults to data/golden/golden.jsonl when no path is given.
Exit code is 0 on success, 1 on failure — so it's CI-friendly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raggate.dataset.loader import DatasetError, load_cases

DEFAULT_PATH = Path("data/golden/golden.jsonl")


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        cases = load_cases(path)
    except DatasetError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for c in cases:
        by_category[c.category.value] = by_category.get(c.category.value, 0) + 1
        by_difficulty[c.difficulty.value] = (
            by_difficulty.get(c.difficulty.value, 0) + 1
        )

    print(f"[OK] {path}")
    print(f"  cases: {len(cases)}")
    print("  by category:")
    for k in sorted(by_category):
        print(f"    {k:<12} {by_category[k]}")
    print("  by difficulty:")
    for k in sorted(by_difficulty):
        print(f"    {k:<12} {by_difficulty[k]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raggate-dataset",
        description="Inspect and validate the RAGGate AI golden dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Validate a golden set JSONL file.")
    p_validate.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help=f"Path to golden JSONL (default: {DEFAULT_PATH})",
    )
    p_validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())