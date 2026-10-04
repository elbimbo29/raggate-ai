"""Tests for the fake judge.

Hand-computed word overlaps. If a test fails, verify the arithmetic
by hand before touching the fake.
"""

from raggate.judge.fake import FakeJudge


def _judge() -> FakeJudge:
    return FakeJudge()


# ---------- faithfulness ----------

def test_faithfulness_all_answer_words_in_context():
    # answer tokens = {rate, limit, is, 100}
    # context tokens = {rate, limit, is, 100, per, minute}
    # overlap = 4/4 = 1.0
    r = _judge().faithfulness(
        question="ignored",
        answer="rate limit is 100",
        context=["the rate limit is 100 per minute"],
    )
    assert r.score == 1.0


def test_faithfulness_half_overlap():
    # answer tokens = {rate, limit, is, 100}
    # context tokens = {rate, limit, per, minute}
    # overlap = 2/4 = 0.5
    r = _judge().faithfulness(
        question="ignored",
        answer="rate limit is 100",
        context=["rate limit per minute"],
    )
    assert r.score == 0.5


def test_faithfulness_empty_answer():
    r = _judge().faithfulness("q", "", ["some context"])
    assert r.score == 0.0
    assert r.reason == "empty answer"


# ---------- answer_relevancy ----------

def test_answer_relevancy_full_overlap():
    # question tokens = {what, is, the, rate, limit}
    # answer tokens = {the, rate, limit, is, 100}
    # overlap = {the, rate, limit, is} = 4/5 = 0.8
    r = _judge().answer_relevancy(
        question="what is the rate limit",
        answer="the rate limit is 100",
    )
    assert r.score == 0.8


def test_answer_relevancy_no_overlap():
    r = _judge().answer_relevancy(
        question="what is the rate limit",
        answer="purple monkey dishwasher",
    )
    assert r.score == 0.0


def test_answer_relevancy_empty_question():
    r = _judge().answer_relevancy("", "anything")
    assert r.score == 0.0
    assert r.reason == "empty question"


# ---------- answer_correctness ----------

def test_answer_correctness_exact_words():
    # reference tokens = {100, requests, per, minute}
    # answer tokens = {100, requests, per, minute}
    # overlap = 4/4 = 1.0
    r = _judge().answer_correctness(
        question="ignored",
        answer="100 requests per minute",
        reference="100 requests per minute",
    )
    assert r.score == 1.0


def test_answer_correctness_partial():
    # reference tokens = {100, requests, per, minute}
    # answer tokens = {100, requests}
    # overlap = 2/4 = 0.5
    r = _judge().answer_correctness(
        question="ignored",
        answer="100 requests",
        reference="100 requests per minute",
    )
    assert r.score == 0.5


def test_answer_correctness_empty_reference():
    r = _judge().answer_correctness("q", "a", "")
    assert r.score == 0.0
    assert r.reason == "empty reference"