"""RAGAS cross-check.

Runs the same generation outputs through both the DeepEval judge and
the RAGAS judge, and reports side-by-side scores. The point is not to
declare a winner but to see whether two independent implementations
agree — evidence that both are measuring something real.
"""

from __future__ import annotations

from pydantic import BaseModel

from raggate.dataset.models import EvalCase
from raggate.generator.base import Generator
from raggate.judge.base import Judge
from raggate.retriever.base import Retriever
from raggate.retriever.corpus import Chunk


class CrossCheckCase(BaseModel):
    case_id: str
    question: str
    answer: str
    deepeval_faithfulness: float
    ragas_faithfulness: float
    deepeval_relevancy: float
    ragas_relevancy: float


class CrossCheckReport(BaseModel):
    n_cases: int
    deepeval_faithfulness_mean: float
    ragas_faithfulness_mean: float
    deepeval_relevancy_mean: float
    ragas_relevancy_mean: float
    cases: list[CrossCheckCase]


def run_cross_check(
    cases: list[EvalCase],
    corpus: list[Chunk],
    retriever: Retriever,
    generator: Generator,
    deepeval_judge: Judge,
    ragas_judge: Judge,
    k: int = 5,
) -> CrossCheckReport:
    """Run each case through both judges and compare scores."""
    chunk_lookup = {c.chunk_id: c for c in corpus}
    results: list[CrossCheckCase] = []

    for case in cases:
        retrieved_ids = retriever.retrieve(case.question, k=k)
        context_chunks = [chunk_lookup[cid] for cid in retrieved_ids if cid in chunk_lookup]
        context_texts = [c.text for c in context_chunks]
        answer = generator.generate(case.question, context_chunks)

        de_faith = deepeval_judge.faithfulness(case.question, answer, context_texts)
        ra_faith = ragas_judge.faithfulness(case.question, answer, context_texts)
        de_relev = deepeval_judge.answer_relevancy(case.question, answer)
        ra_relev = ragas_judge.answer_relevancy(case.question, answer)

        results.append(
            CrossCheckCase(
                case_id=case.id,
                question=case.question,
                answer=answer,
                deepeval_faithfulness=de_faith.score,
                ragas_faithfulness=ra_faith.score,
                deepeval_relevancy=de_relev.score,
                ragas_relevancy=ra_relev.score,
            )
        )

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    return CrossCheckReport(
        n_cases=len(results),
        deepeval_faithfulness_mean=_mean([r.deepeval_faithfulness for r in results]),
        ragas_faithfulness_mean=_mean([r.ragas_faithfulness for r in results]),
        deepeval_relevancy_mean=_mean([r.deepeval_relevancy for r in results]),
        ragas_relevancy_mean=_mean([r.ragas_relevancy for r in results]),
        cases=results,
    )