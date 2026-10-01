# RAGGate AI

**An automated evaluation harness and regression gate for production RAG pipelines.**

RAGGate AI scores the retrieval and generation quality of a RAG pipeline against a golden dataset, then enforces a regression gate in CI — so quality drops are caught before they ship.

> **Status:** Phase 0 complete — project scaffolding. Evaluation logic lands in Phase 1.

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
| 1 | Dataset & golden set | ⏳ Next |
| 2 | Retrieval scoring | ⏳ |
| 3 | Generation scoring | ⏳ |
| 4 | FastAPI service | ⏳ |
| 5 | Regression gate | ⏳ |
| 6 | Dashboard & observability | ⏳ |
| 7 | CI integration, polish, demo | ⏳ |

---

## What works today

The API boots and exposes a health endpoint: 
GET /health → {"status": "ok", "version": "0.1.0", "env": "dev"}

---

raggate-ai/
├── src/raggate/       # application code
├── tests/             # test suite
├── docs/              # images and diagrams (real files, not Mermaid)
├── data/              # local SQLite (gitignored)
├── reports/           # eval run outputs (gitignored)
├── scripts/           # one-off helpers
├── Makefile
└── pyproject.toml

---

### 3. Fix `.gitignore` — commit `uv.lock`

Earlier I had you put `uv.lock` in `.gitignore`. That was wrong for an application repo. Remove this line from `.gitignore`:

```gitignore
# uv
uv.lock