# RAGGate AI

**An automated evaluation harness and regression gate for production RAG pipelines.**

RAGGate AI scores the retrieval and generation quality of a RAG pipeline against a golden dataset, then enforces a regression gate in CI — so quality drops are caught before they ship.

> **Status:** Phase 1 complete — golden dataset and loader. Retrieval scoring lands in Phase 2.

---

## Why

RAG pipelines degrade silently. A prompt tweak, an embedding model swap, or a chunker change can quietly tank answer quality without breaking a single test. RAGGate AI turns "does this RAG pipeline still work?" into a measurable, enforceable signal:

- **Scores** retrieval (hit-rate@k, MRR, context precision/recall) and generation (faithfulness, answer relevancy, correctness).
- **Gates** quality regressions by comparing a candidate run to a baseline against configurable thresholds.
- **Blocks** merges in CI when quality drops.

---

## Status

| Phase | Description | State |
|---|---|---|
| 0 | Project skeleton & foundation | ✅ Done |
| 1 | Dataset & golden set | ✅ Done |
| 2 | Retrieval scoring | ⏳ Next |
| 3 | Generation scoring | ⏳ |
| 4 | FastAPI service | ⏳ |
| 5 | Regression gate | ⏳ |
| 6 | Dashboard & observability | ⏳ |
| 7 | CI integration, polish, demo | ⏳ |

---

## What works today

- **FastAPI service** boots with a health endpoint:
  `GET /health` → `{"status": "ok", "version": "0.1.0", "env": "dev"}`
- **Golden dataset** with 20 curated cases, validated by a CLI.

---

## The Dataset

RAGGate AI evaluates against a **golden set**: a curated list of questions, each with a ground-truth answer and the corpus chunks that should be retrieved to answer it.

Two files under `data/`:

- **`data/corpus/acmedb.jsonl`** — the retrieval corpus. 18 short doc chunks for *AcmeDB*, a fictional managed SQL database with a REST API.
- **`data/golden/golden.jsonl`** — 20 golden cases spread across six categories: `auth`, `endpoints`, `data_model`, `limits`, `errors`, and `misc`.

Each golden case has this shape:

```json
{
  "id": "errors-004",
  "question": "Why am I getting a 429, and what should I do about it?",
  "expected_answer": "You have exceeded the 100 requests-per-minute rate limit. Wait and retry with exponential backoff (base 500 ms, max 5 retries).",
  "expected_chunk_ids": ["chunk-limits-01", "chunk-errors-03"],
  "category": "errors",
  "difficulty": "hard",
  "notes": "Cross-cutting: needs both the limits chunk and the retry-guidance chunk."
}
```

Most cases reference a single expected chunk. A few are deliberately **cross-cutting**, referencing two chunks, so retrieval metrics have to handle multi-source ground truth.

### Validating the golden set

```bash
uv run raggate-dataset validate
```

![Golden set validation](docs/phase1-dataset-validate.png)

The loader fails loudly and precisely — bad lines are reported with their line number and the specific schema violation, and the command exits non-zero so it's CI-friendly.

---

## Stack

| Concern | Choice |
|---|---|
| Package manager | uv |
| API | FastAPI + Uvicorn |
| Settings | pydantic-settings |
| Data validation | Pydantic v2 |
| Tests | pytest |
| Lint / format | ruff |
| Storage (planned) | SQLite |
| Evaluation (planned) | DeepEval, RAGAS |
| Dashboard (planned) | Streamlit |
| CI (planned) | GitHub Actions |

---

## Quickstart

```bash
# Clone
git clone git@github.com:elbimbo29/raggate-ai.git
cd raggate-ai

# Install
uv sync --extra dev

# Run tests
make test

# Validate the golden set
uv run raggate-dataset validate

# Boot the API
make dev
# → http://127.0.0.1:8000/docs
```

---

## Layout

```
raggate-ai/
├── src/raggate/          # application code
│   ├── dataset/          # Phase 1 — golden set + corpus
│   ├── main.py           # FastAPI app
│   └── config.py         # settings
├── tests/                # test suite
├── docs/                 # screenshots (real files, no Mermaid)
├── data/
│   ├── corpus/           # AcmeDB source documents
│   └── golden/           # golden dataset
├── reports/              # eval outputs (gitignored)
├── scripts/              # one-off helpers
├── Makefile
└── pyproject.toml
```

---

## License

MIT