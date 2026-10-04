"""CLI for running retrieval evaluation.

Usage:
    raggate-retrieval run --retriever keyword --k 5
    raggate-retrieval run --retriever chroma --k 5 --out reports/foo.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raggate.dataset.loader import DatasetError, load_cases
from raggate.harness.retrieval_runner import run_retrieval
from raggate.retriever.chroma import ChromaRetriever
from raggate.retriever.corpus import CorpusError, load_corpus
from raggate.retriever.keyword import KeywordRetriever

DEFAULT_CORPUS = Path("data/corpus/acmedb.jsonl")
DEFAULT_GOLDEN = Path("data/golden/golden.jsonl")
DEFAULT_OUT_DIR = Path("reports")

RETRIEVERS = {
    "keyword": KeywordRetriever,
    "chroma": ChromaRetriever,
}


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.golden)
        chunks = load_corpus(args.corpus)
    except (DatasetError, CorpusError) as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    retriever_cls = RETRIEVERS[args.retriever]
    retriever = retriever_cls(chunks)

    report = run_retrieval(cases, retriever, k=args.k)

    print(f"[OK] {report.retriever_name}  cases={report.n_cases}  k={report.k}")
    for name, value in report.metrics.items():
        print(f"  {name:<22} {value:.4f}")

    if args.out:
        out_path = Path(args.out)
    else:
        # Default path encodes retriever and k so runs don't clobber each other.
        out_path = DEFAULT_OUT_DIR / f"retrieval-{report.retriever_name}-k{args.k}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"  saved: {out_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raggate-retrieval",
        description="Run retrieval evaluation against the golden set.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run retrieval and print + save a report.")
    p_run.add_argument(
        "--retriever",
        choices=sorted(RETRIEVERS),
        default="keyword",
        help="Which retriever to evaluate.",
    )
    p_run.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics.")
    p_run.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    p_run.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p_run.add_argument("--out", default=None, help="Output JSON path.")
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())