"""Run the retrieval eval on the current codebase and gate it against
the committed baseline.

Used by the eval workflow in CI. Prints a human-readable verdict to
stdout and writes a machine-readable result to /tmp/eval-result.json
for the workflow to read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from raggate.dataset.loader import load_cases
from raggate.gate.config import load_thresholds
from raggate.gate.evaluator import evaluate_gate
from raggate.harness.retrieval_runner import run_retrieval
from raggate.retriever.chroma import ChromaRetriever
from raggate.retriever.corpus import load_corpus
from raggate.retriever.keyword import KeywordRetriever

BASELINE_PATH = Path("config/baseline.json")
GOLDEN_PATH = Path("data/golden/golden.jsonl")
CORPUS_PATH = Path("data/corpus/acmedb.jsonl")
RESULT_PATH = Path("/tmp/eval-result.json")


def main() -> int:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    kind = baseline["kind"]
    k = baseline["k"]

    print(f"== CI eval: {kind} baseline ==")
    print(f"  retriever={baseline['retriever']}  k={k}  n_cases={baseline['n_cases']}")

    cases = load_cases(GOLDEN_PATH)
    corpus = load_corpus(CORPUS_PATH)
    retriever = KeywordRetriever(corpus)

    report = run_retrieval(cases, retriever, k=k)

    print()
    print("  current metrics:")
    for name, value in report.metrics.items():
        base = baseline["metrics"].get(name)
        if base is None:
            print(f"    {name:<22} {value:.4f}")
        else:
            delta = value - base
            print(
                f"    {name:<22} {value:.4f}  "
                f"(baseline {base:.4f}, delta {delta:+.4f})"
            )

    thresholds = load_thresholds(kind)
    result = evaluate_gate(
        baseline_metrics=baseline["metrics"],
        candidate_metrics=report.metrics,
        thresholds=thresholds,
    )

    print()
    print(f"  gate: {'PASS' if result.passed else 'FAIL'}")
    for reg in result.regressions:
        print(
            f"    * {reg.metric:<22} "
            f"baseline={reg.baseline:.4f}  "
            f"candidate={reg.candidate:.4f}  "
            f"delta={reg.delta:+.4f}  "
            f"threshold={reg.threshold:.4f}"
        )

    # Machine-readable result for the workflow to consume.
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "passed": result.passed,
                "regressions": [r.model_dump() for r in result.regressions],
                "improvements": [i.model_dump() for i in result.improvements],
                "baseline_metrics": baseline["metrics"],
                "candidate_metrics": report.metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())