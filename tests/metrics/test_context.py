"""Tests for context precision and context recall.

Hand-computed arithmetic in every comment. If a test fails, do the math
by hand before changing the code.
"""

import pytest

from raggate.metrics.context import context_precision, context_recall

# ---------- context_precision ----------

def test_context_precision_perfect():
    # expected = {A}, retrieved = [A, X, Y]. Only A relevant at rank 1.
    # precision@1 = 1/1 = 1.0. Average over [1.0] = 1.0.
    assert context_precision({"A"}, ["A", "X", "Y"], k=3) == 1.0


def test_context_precision_with_one_intruder():
    # expected = {A, C}, retrieved = [A, X, C, Y].
    # i=1: A relevant -> precision@1 = 1/1 = 1.0
    # i=3: C relevant -> precision@3 = 2/3 ≈ 0.6667
    # Mean = (1.0 + 0.6667) / 2 = 0.8333
    assert context_precision({"A", "C"}, ["A", "X", "C", "Y"], k=4) == pytest.approx(
        (1.0 + 2 / 3) / 2
    )


def test_context_precision_relevant_only_at_bottom():
    # expected = {C}, retrieved = [X, Y, C].
    # i=3: C relevant -> precision@3 = 1/3 ≈ 0.3333
    # Mean = 0.3333
    assert context_precision({"C"}, ["X", "Y", "C"], k=3) == pytest.approx(1 / 3)


def test_context_precision_no_relevant_in_top_k():
    # expected = {C}, retrieved = [X, Y, C], k=2 -> C is out of top-2
    # No relevant chunks seen -> 0.0
    assert context_precision({"C"}, ["X", "Y", "C"], k=2) == 0.0


def test_context_precision_empty_retrieval():
    assert context_precision({"A"}, [], k=3) == 0.0


def test_context_precision_rejects_non_positive_k():
    with pytest.raises(ValueError):
        context_precision({"A"}, ["A"], k=0)


# ---------- context_recall ----------

def test_context_recall_all_found():
    # expected = {A, B}, both in top-3 -> 2/2 = 1.0
    assert context_recall({"A", "B"}, ["A", "B", "C"], k=3) == 1.0


def test_context_recall_partial():
    # expected = {A, B}, only A found in top-3 -> 1/2 = 0.5
    assert context_recall({"A", "B"}, ["A", "X", "Y"], k=3) == 0.5


def test_context_recall_none_found():
    assert context_recall({"A", "B"}, ["X", "Y"], k=3) == 0.0


def test_context_recall_respects_k():
    # expected = {C}, retrieved = [X, Y, C], k=2 -> miss -> 0.0
    assert context_recall({"C"}, ["X", "Y", "C"], k=2) == 0.0


def test_context_recall_empty_expected():
    assert context_recall(set(), ["A"], k=3) == 0.0


def test_context_recall_rejects_non_positive_k():
    with pytest.raises(ValueError):
        context_recall({"A"}, ["A"], k=0)