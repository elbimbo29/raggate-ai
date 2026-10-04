"""CLI for running generation evaluation.

Usage:
    raggate-generation run --retriever chroma --generator template --judge deepeval
    raggate-generation run --retriever chroma --judge ragas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raggate.dataset.loader import DatasetError, load_cases
from raggate.generator.template import TemplateGenerator
from raggate.harness.generation_runner import run_generation
from raggate.judge.deepeval_judge import DeepEvalJudge
from raggate.judge.ragas_judge import RagasJudge
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

GENERATORS = {
    "template": TemplateGenerator,
}

JUDGES = {
    "deepeval": DeepEvalJudge,
    "ragas": RagasJudge,
}


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.golden)
        corpus = load_corpus(args.corpus)
    except (DatasetError, CorpusError) as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    retriever = RETRIEVERS[args.retriever](corpus)
    generator = GENERATORS[args.generator]()
    judge = JUDGES[args.judge]()

    report = run_generation(
        cases, corpus, retriever, generator, judge, k=args.k
    )

    print(
        f"[OK] retriever={report.retriever_name}  "
        f"generator={report.generator_name}  "
        f"judge={report.judge_name}"
    )
    print(f"  cases={report.n_cases}  k={report.k}")
    print(
        f"  cost_usd={report.cost_usd_total:.4f}  "
        f"latency_ms={report.latency_ms_total:.0f}"
    )
    for name, value in report.metrics.items():
        print(f"  {name:<22} {value:.4f}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (
            DEFAULT_OUT_DIR
            / f"generation-{args.retriever}-{args.generator}-{args.judge}-k{args.k}.json"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"  saved: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raggate-generation",
        description="Run generation evaluation with an LLM judge.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run generation and save a report.")
    p_run.add_argument(
        "--retriever", choices=sorted(RETRIEVERS), default="chroma"
    )
    p_run.add_argument(
        "--generator", choices=sorted(GENERATORS), default="template"
    )
    p_run.add_argument("--judge", choices=sorted(JUDGES), default="deepeval")
    p_run.add_argument("--k", type=int, default=5)
    p_run.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    p_run.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p_run.add_argument("--out", default=None)
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())