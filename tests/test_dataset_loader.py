"""Tests for the golden-set loader."""


import json
from pathlib import Path

import pytest

from raggate.dataset.loader import DatasetError, load_cases

REPO_ROOT = Path(__file__).resolve().parents[1]



def _write(tmp_path, lines: list[str]):
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _case(id_: str = "auth-001", **overrides) -> dict:
    base = {
        "id": id_,
        "question": "Q?",
        "expected_answer": "A.",
        "expected_chunk_ids": ["chunk-auth-01"],
        "category": "auth",
        "difficulty": "easy",
    }
    base.update(overrides)
    return base


def test_loads_valid_file(tmp_path):
    p = _write(tmp_path, [json.dumps(_case("auth-001")), json.dumps(_case("auth-002"))])
    cases = load_cases(p)
    assert [c.id for c in cases] == ["auth-001", "auth-002"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_cases(tmp_path / "nope.jsonl")


def test_empty_file_raises(tmp_path):
    p = _write(tmp_path, [])
    with pytest.raises(DatasetError, match="empty"):
        load_cases(p)


def test_blank_and_comment_lines_ignored(tmp_path):
    p = _write(tmp_path, ["", "# a comment", json.dumps(_case())])
    cases = load_cases(p)
    assert len(cases) == 1


def test_invalid_json_reports_line_number(tmp_path):
    p = _write(tmp_path, [json.dumps(_case()), "{not json"])
    with pytest.raises(DatasetError, match=r"line 2: invalid JSON"):
        load_cases(p)


def test_schema_violation_reports_line_number(tmp_path):
    bad = _case()
    bad["category"] = "not_a_category"
    p = _write(tmp_path, [json.dumps(_case()), json.dumps(bad)])
    with pytest.raises(DatasetError, match=r"line 2"):
        load_cases(p)


def test_duplicate_ids_rejected(tmp_path):
    p = _write(tmp_path, [json.dumps(_case("auth-001")), json.dumps(_case("auth-001"))])
    with pytest.raises(DatasetError, match="duplicate id 'auth-001'"):
        load_cases(p)


def test_multiple_errors_all_reported(tmp_path):
    p = _write(
        tmp_path,
        [
            json.dumps(_case("auth-001")),
            "{bad json",
            json.dumps(_case("auth-001")),  # duplicate
        ],
    )
    with pytest.raises(DatasetError) as exc:
        load_cases(p)
    msg = str(exc.value)
    assert "line 2" in msg
    assert "line 3" in msg


def test_shipped_golden_set_is_valid():
    """The golden set committed to the repo must always validate."""
    cases = load_cases(REPO_ROOT / "data" / "golden" / "golden.jsonl")
    assert len(cases) == 20
    assert len({c.id for c in cases}) == 20


def test_shipped_golden_set_chunks_exist_in_corpus():
    """Every expected_chunk_id must resolve to a chunk in the corpus."""
    corpus_ids = {
        json.loads(line)["chunk_id"]
        for line in (REPO_ROOT / "data" / "corpus" / "acmedb.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    for case in load_cases(REPO_ROOT / "data" / "golden" / "golden.jsonl"):
        for cid in case.expected_chunk_ids:
            assert cid in corpus_ids, f"{case.id} references missing chunk {cid}"    