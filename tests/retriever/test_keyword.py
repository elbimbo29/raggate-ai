"""Tests for the keyword baseline retriever."""

from raggate.retriever.corpus import Chunk
from raggate.retriever.keyword import KeywordRetriever, _tokenize


def _chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="c1", section="a", text="The rate limit is 100 per minute."),
        Chunk(chunk_id="c2", section="a", text="Authenticate with a Bearer token."),
        Chunk(chunk_id="c3", section="b", text="Available in three regions."),
    ]


def test_tokenize_lowercases_and_splits():
    assert _tokenize("The Rate-Limit is 100/min.") == {"the", "rate", "limit", "is", "100", "min"}


def test_retriever_returns_top_overlap_first():
    r = KeywordRetriever(_chunks())
    assert r.retrieve("rate limit", k=3)[0] == "c1"


def test_retriever_respects_k():
    r = KeywordRetriever(_chunks())
    assert len(r.retrieve("the rate limit is", k=1)) == 1


def test_retriever_skips_zero_overlap_chunks():
    r = KeywordRetriever(_chunks())
    # "xyzzy" matches nothing
    assert r.retrieve("xyzzy", k=3) == []


def test_retriever_deterministic_tiebreak():
    chunks = [
        Chunk(chunk_id="z", section="a", text="alpha"),
        Chunk(chunk_id="a", section="a", text="alpha"),
    ]
    r = KeywordRetriever(chunks)
    # Equal overlap -> sorted by chunk_id
    assert r.retrieve("alpha", k=2) == ["a", "z"]


def test_retriever_name_is_configurable():
    r = KeywordRetriever(_chunks(), name="kw-test")
    assert r.name == "kw-test"