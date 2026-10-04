"""Retrieval metrics.

All metrics operate on a single (case, retrieval result) pair:

  - expected: set[str]  — chunk IDs the case says SHOULD be retrieved
  - retrieved: list[str] — chunk IDs the retriever returned, best-first

None of these look at text or embeddings; they only compare ID sets and
rankings. That keeps them fast, deterministic, and easy to reason about.
"""

from __future__ import annotations


def hit_rate_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    """1.0 if any expected chunk is in the top-k retrieved, else 0.0.

    Binary per case. Averaged across cases it answers: "on what fraction
    of questions did the retriever find at least one correct source?"
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return 1.0 if set(retrieved[:k]) & expected else 0.0


def mrr(expected: set[str], retrieved: list[str]) -> float:
    """Reciprocal rank of the first expected chunk in the ranking.

    If the first expected chunk is at rank 1, score is 1.0.
    At rank 2, 0.5. At rank 3, 0.333. If no expected chunk appears,
    score is 0.0. No `k` — MRR considers the full ranking.
    """
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of expected chunks that appear in the top-k retrieved.

    Differs from hit_rate: hit_rate only asks whether ANY correct chunk
    was found; recall asks WHICH FRACTION of the correct chunks were
    found. A case with two expected chunks that retrieves one of them
    scores hit_rate=1.0 but recall=0.5.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not expected:
        return 0.0
    return len(set(retrieved[:k]) & expected) / len(expected)