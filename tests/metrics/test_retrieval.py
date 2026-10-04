"""Tests for retrieval metrics.

Each test includes the hand-computed arithmetic in a comment.
If a test fails, verify the math by hand before "fixing" the code.
"""

import pytest

from raggate.metrics.retrieval import hit_rate_at_k, mrr, recall_at_k

# ---------- hit_rate_at_k ----------

def test_hit_rate_hit_at_rank_1():
    # retrieved[0] is expected -> hit -> 1.0
    assert hit_rate_at_k({"a"}, ["a", "b", "c"], k=3) == 1.0


def test_hit_rate_hit_at_rank_3():
    # expected 'c' is at index 2, within top-3 -> 1.0
    assert hit_rate_at_k({"c"}, ["a", "b", "c"], k=3) == 1.0


def test_hit_rate_miss_because_out_of_top_k():
    # expected 'c' at index 2, but k=2 -> only ['a','b'] considered -> 0.0
    assert hit_rate_at_k({"c"}, ["a", "b", "c"], k=2) == 0.0


def test_hit_rate_empty_retrieval():
    assert hit_rate_at_k({"a"}, [], k=3) == 0.0


def test_hit_rate_rejects_non_positive_k():
    with pytest.raises(ValueError):
        hit_rate_at_k({"a"}, ["a"], k=0)


# ---------- mrr ----------

def test_mrr_first_expected_at_rank_1():
    # rank 1 -> 1/1 = 1.0
    assert mrr({"a"}, ["a", "b", "c"]) == 1.0


def test_mrr_first_expected_at_rank_2():
    # rank 2 -> 1/2 = 0.5
    assert mrr({"b"}, ["a", "b", "c"]) == 0.5


def test_mrr_first_expected_at_rank_3():
    # rank 3 -> 1/3 ≈ 0.3333
    assert mrr({"c"}, ["a", "b", "c"]) == pytest.approx(1 / 3)


def test_mrr_no_match():
    assert mrr({"x"}, ["a", "b", "c"]) == 0.0


def test_mrr_uses_first_hit_not_best():
    # expected = {'b', 'c'}. 'b' is at rank 2, 'c' at rank 3.
    # MRR uses the FIRST match, so 1/2 = 0.5, not 1/3.
    assert mrr({"b", "c"}, ["a", "b", "c"]) == 0.5


# ---------- recall_at_k ----------

def test_recall_single_expected_found():
    # expected = {'a'}, retrieved top-3 contains 'a' -> 1/1 = 1.0
    assert recall_at_k({"a"}, ["a", "b", "c"], k=3) == 1.0


def test_recall_half_of_two_expected():
    # expected = {'a', 'c'}. top-3 contains 'a' but not 'c' -> 1/2 = 0.5
    assert recall_at_k({"a", "c"}, ["a", "b", "d"], k=3) == 0.5


def test_recall_all_of_three_expected():
    # expected = {'a', 'b', 'c'}. top-3 = all three -> 3/3 = 1.0
    assert recall_at_k({"a", "b", "c"}, ["a", "b", "c"], k=3) == 1.0


def test_recall_bounded_by_k():
    # expected = {'c'}, retrieved at rank 5 with k=3 -> miss -> 0.0
    assert recall_at_k({"c"}, ["a", "b", "x", "y", "c"], k=3) == 0.0


def test_recall_empty_expected_returns_zero():
    assert recall_at_k(set(), ["a", "b"], k=3) == 0.0


def test_recall_rejects_non_positive_k():
    with pytest.raises(ValueError):
        recall_at_k({"a"}, ["a"], k=0)