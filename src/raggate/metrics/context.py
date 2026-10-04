
"""Context precision and recall.

These are RAGAS-style names for two rank-aware retrieval metrics. With
ID-based ground truth (our setup), context_recall is mathematically the
same as recall@k. context_precision is genuinely different: it weights
relevant chunks by how early they appear in the ranking.
"""

from __future__ import annotations


def context_precision(expected: set[str], retrieved: list[str], k: int) -> float:
    """RAGAS-style mean precision across relevant chunks, top-k only."""
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = retrieved[:k]
    precisions: list[float] = []
    relevant_seen = 0

    for i, chunk_id in enumerate(top_k, start=1):
        if chunk_id in expected:
            relevant_seen += 1
            precisions.append(relevant_seen / i)

    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def context_recall(expected: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of expected chunks found in the top-k retrieved."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not expected:
        return 0.0
    return len(set(retrieved[:k]) & expected) / len(expected)
