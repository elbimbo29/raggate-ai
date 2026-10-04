"""Tests for the template generator."""

from raggate.generator.template import MAX_CHARS, TemplateGenerator
from raggate.retriever.corpus import Chunk


def _chunk(id_: str, text: str) -> Chunk:
    return Chunk(chunk_id=id_, section="test", text=text)


def test_template_joins_context_in_order():
    g = TemplateGenerator()
    ctx = [_chunk("a", "First sentence."), _chunk("b", "Second sentence.")]
    assert g.generate("ignored question", ctx) == "First sentence. Second sentence."


def test_template_ignores_question():
    g = TemplateGenerator()
    ctx = [_chunk("a", "Only content.")]
    # Same context, different question -> same answer.
    assert g.generate("q1", ctx) == g.generate("q2", ctx)


def test_template_empty_context_returns_empty_string():
    g = TemplateGenerator()
    assert g.generate("q", []) == ""


def test_template_truncates_long_context():
    g = TemplateGenerator()
    long_text = "word " * (MAX_CHARS)  # definitely over the limit
    ctx = [_chunk("a", long_text)]
    out = g.generate("q", ctx)
    assert len(out) <= MAX_CHARS + len("...")
    assert out.endswith("...")


def test_template_name_is_configurable():
    g = TemplateGenerator(name="tpl-test")
    assert g.name == "tpl-test"