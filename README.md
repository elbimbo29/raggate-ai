# RAGGate AI

**An automated evaluation harness and regression gate for production RAG pipelines.**

RAGGate AI scores the retrieval and generation quality of a RAG pipeline against a golden dataset, then enforces a regression gate in CI — so quality drops are caught before they ship.

> **Status:** Phase 4 complete — HTTP service with run persistence and comparison. Regression gate lands in Phase 5.

---

## Why

RAG pipelines degrade silently. A prompt tweak, an embedding model swap, or a chunker change can quietly tank answer quality without breaking a single test. RAGGate AI turns "does this RAG pipeline still work?" into a measurable, enforceable signal:

- **Scores** retrieval (hit-rate@k, MRR, context precision/recall) and generation (faithfulness, answer relevancy, correctness).
- **Gates** quality regressions by comparing a candidate run to a baseline against configurable thresholds.
- **Blocks** merges in CI when quality drops.

---

## Status

| 0 | Project skeleton & foundation | ✅ Done |
| 1 | Dataset & golden set | ✅ Done |
| 2 | Retrieval scoring | ✅ Done |
| 3 | Generation scoring | ✅ Done |
| 4 | FastAPI service | ✅ Done |
| 5 | Regression gate | ⏳ Next |
| 6 | Dashboard & observability | ⏳ |
| 7 | CI integration, polish, demo | ⏳ |
---

## What works today

- **Golden dataset** with 20 curated cases over an 18-chunk AcmeDB corpus, validated by a CLI.
- **Five retrieval metrics** — hit-rate@k, MRR, recall@k, context precision, context recall — with hand-computed tests.
- **Three generation metrics** — faithfulness, answer relevancy, answer correctness — scored by DeepEval and cross-checked against RAGAS.
- **Two retrievers** — keyword baseline (negative control) and Chroma + MiniLM embeddings.
- **HTTP service** with run persistence, background execution, list/fetch, and side-by-side comparison.
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

## Retrieval Metrics

RAGGate AI scores retrieval with five metrics. All are computed from ID-based ground truth (`expected_chunk_ids` in the golden set) — no LLM judge involved, so they're fast, deterministic, and free.

| Metric | What it asks | Range |
|---|---|---|
| **hit-rate@k** | On what fraction of questions did we find at least one correct chunk in the top k? | 0.0 – 1.0 |
| **MRR** | Where did the *first* correct chunk land in the ranking? | 0.0 – 1.0 |
| **recall@k** | What fraction of the expected chunks made it into the top k? | 0.0 – 1.0 |
| **context precision** | Of the chunks we retrieved, how many were relevant? (rank-weighted) | 0.0 – 1.0 |
| **context recall** | Same as recall@k, named RAGAS-style for cross-checking. | 0.0 – 1.0 |

### Two retrievers, one negative control

The harness ships with two retrievers so the metrics can be validated against a known-good and a known-weak baseline:

- **`keyword`** — token-overlap baseline. Deterministic, no dependencies, intentionally weak. Acts as a **negative control**: if the harness can't detect this retriever's failures, the harness is broken.
- **`chroma`** — sentence-embedding retriever over the same corpus (`all-MiniLM-L6-v2`, cosine similarity). Should clearly beat keyword.

```bash
uv run raggate-retrieval run --retriever keyword --k 5
uv run raggate-retrieval run --retriever chroma  --k 5
```

![Retrieval comparison](docs/phase2-retrieval-comparison.png)

Two things the comparison shows:

1. **The keyword baseline fails visibly.** Questions like *"How do I authenticate?"* return zero chunks — the corpus says *"authenticates"*, and token overlap misses it. The metrics catch this: `hit_rate@5` and `context_precision` drop meaningfully.
2. **The embedding retriever closes the gap.** Same corpus, same golden set, same metrics — only the retrieval method changed. That's the harness doing its job.

Every run writes a JSON report to `reports/`, containing both the aggregate metrics and the per-case detail. Those JSON files are what the regression gate (Phase 5) will compare.
---

## Generation Metrics

Retrieval metrics tell you whether the *right chunks* came back. Generation metrics tell you whether the *answer* is any good. These require an **LLM judge** — a model that reads the answer alongside the question and context and scores it.

RAGGate AI supports three generation metrics via two independent judge backends (DeepEval and RAGAS), with `gpt-4o-mini` as the judge model.

| Metric | What it asks |
|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? (Catches hallucination.) |
| **Answer relevancy** | Does the answer actually address the question? |
| **Answer correctness** | Does the answer match the reference answer? (DeepEval only — RAGAS 0.4 dropped this metric.) |

Every judge call tracks **cost** and **latency**, because in production those matter as much as the score. A full 20-case run with three generation metrics takes roughly **7 minutes** and costs **~$0.01** with `gpt-4o-mini`.

### Two judges, one cross-check

Because LLM-judged metrics are *approximations* rather than ground truth, RAGGate AI ships with two independent judge backends — DeepEval and RAGAS — and runs the same inputs through both:

| Metric | DeepEval | RAGAS | Δ |
|---|---|---|---|
| Faithfulness | 0.9678 | 0.9773 | −0.0095 |
| Answer relevancy | 0.3754 | 0.7274 | −0.3520 |

**Faithfulness agrees almost exactly** — both frameworks operationalize "supported by context" the same way, and the scores confirm the template generator is highly faithful (it copies context verbatim).

**Relevancy diverges by 0.35.** Both frameworks are "right" — they just define relevancy differently. DeepEval generates candidate questions from the answer and scores how well they match the original question; RAGAS uses synthetic-question cosine similarity. The template generator ignores the question entirely, so a stricter relevancy metric sees it as less relevant than a lenient one does.

The honest takeaway: **an LLM-judged score is only meaningful alongside the rubric that produced it.** Cross-checking against a second framework is cheap and worth doing.

![Cross-check and generation metrics](docs/phase3-cross-check.png)

### A note on non-determinism

Rerunning the same generation evaluation produces slightly different scores. In two back-to-back runs, DeepEval's answer relevancy on the same inputs moved from 0.3409 to 0.3754. This is inherent to LLM-as-judge — the model is stochastic. It's why the regression gate (Phase 5) uses **thresholds** rather than exact equality, and why the dashboard (Phase 6) shows run-over-run trends rather than single points.

---
## Service API

Everything above is exposed over HTTP. Runs are persisted to SQLite, and clients can start a run, poll its status, fetch its full report, and compare two runs side by side.

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Service health and version. |
| `/eval/run` | POST | Start an evaluation run. Returns a `run_id` immediately (202). |
| `/eval/runs` | GET | List recent runs, newest first. Filter with `?kind=retrieval\|generation`, cap with `?limit=N`. |
| `/eval/runs/{run_id}` | GET | Fetch a run's summary plus every per-case result. |
| `/eval/compare` | POST | Compare two runs' metrics, return per-metric deltas. |

![OpenAPI docs](docs/phase4-openapi.png)

### Background execution

A generation run takes 5–10 minutes. `POST /eval/run` doesn't block — it inserts a `pending` run record, schedules the work as a background task, and returns the run ID immediately. Clients poll `GET /eval/runs/{id}` to see status transition through `pending → running → succeeded` (or `failed`, with the error message stored on the record).

### Compare in action

The same two retrievers from Phase 2, now compared over HTTP:

```bash
curl -X POST http://127.0.0.1:8000/eval/compare \
  -H "Content-Type: application/json" \
  -d '{"baseline_run_id":"<keyword-run>","candidate_run_id":"<chroma-run>"}'
```

![Compare endpoint](docs/phase4-compare.png)

Every metric moves in the right direction: `hit_rate@5`, `mrr`, `recall@5`, `context_precision`, and `context_recall` all improve when the retriever changes from keyword overlap to embeddings. **This is the seed of the regression gate** — Phase 5 adds thresholds so the endpoint returns a pass/fail verdict instead of raw numbers.

### Persistence

Runs and per-case results are stored in SQLite via a `RunStore` interface. The concrete `SQLiteRunStore` is the only implementation today; a Postgres implementation is a drop-in (the API layer only knows the interface). The database lives at `data/raggate.sqlite` by default, configurable via `RAGGATE_DB_PATH`.
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