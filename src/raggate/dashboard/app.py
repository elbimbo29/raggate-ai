"""RAGGate AI — Streamlit dashboard.

Reads from the same SQLite store the API writes to. No LLM calls, no
cost. Three views: Runs, Compare, Gate.

Run with:
    make dashboard
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from raggate.config import settings
from raggate.gate.config import DEFAULT_PATH as DEFAULT_THRESHOLDS_PATH
from raggate.gate.config import ThresholdsConfigError, load_thresholds
from raggate.gate.evaluator import evaluate_gate
from raggate.storage.sqlite import SQLiteRunStore

st.set_page_config(
    page_title="RAGGate AI",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def get_store(db_path: str) -> SQLiteRunStore:
    """One store per DB path. Cached so we don't reconnect every rerun."""
    return SQLiteRunStore(db_path)


def _runs_dataframe(store: SQLiteRunStore, kind: str | None, limit: int) -> pd.DataFrame:
    records = store.list_runs(kind=kind, limit=limit)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "id": r.id[:12],
                "kind": r.kind,
                "status": r.status,
                "retriever": r.retriever_name,
                "generator": r.generator_name or "—",
                "judge": r.judge_name or "—",
                "k": r.k,
                "n_cases": r.n_cases,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at,
                **r.metrics,
            }
            for r in records
        ]
    )


def _render_runs_view(store: SQLiteRunStore) -> None:
    st.subheader("Runs")

    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        kind = st.selectbox("Filter by kind", ["all", "retrieval", "generation"])
    with col2:
        limit = st.number_input("Max runs", min_value=5, max_value=200, value=20)
    with col3:
        st.write("")  # spacer

    kind_filter = None if kind == "all" else kind
    df = _runs_dataframe(store, kind_filter, int(limit))

    if df.empty:
        st.info("No runs found. Start one with `POST /eval/run`.")
        return

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", len(df))
    c2.metric("Succeeded", int((df["status"] == "succeeded").sum()))
    c3.metric("Failed", int((df["status"] == "failed").sum()))
    c4.metric("Total cost", f"${df['cost_usd'].sum():.4f}")

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-case drill-down
    st.markdown("### Per-case detail")
    run_ids = list(df["id"])
    selected = st.selectbox("Pick a run (showing first 12 chars of id)", run_ids)

    if selected:
        # We stored short IDs for display; look up the full record by prefix.
        records = store.list_runs(kind=kind_filter, limit=int(limit))
        full_id = next((r.id for r in records if r.id.startswith(selected)), None)
        if full_id is None:
            st.warning("Run not found.")
            return

        cases = store.get_run_cases(full_id)
        if not cases:
            st.info("No case-level results stored for this run.")
            return

        cases_df = pd.DataFrame([{"case_id": c.case_id, **c.payload} for c in cases])
        # Drop noisy columns for readability.
        for drop in ("question", "answer", "reference_answer", "reasons"):
            if drop in cases_df.columns:
                cases_df = cases_df.drop(columns=[drop])
        st.dataframe(cases_df, use_container_width=True, hide_index=True)

def _render_compare_view(store: SQLiteRunStore) -> None:
    st.subheader("Compare runs")

    records = store.list_runs(limit=100)
    if len(records) < 2:
        st.info("Need at least two runs to compare. Start more with `POST /eval/run`.")
        return

    labels = {
        f"{r.id[:12]} | {r.kind} | {r.retriever_name} | {r.created_at}": r.id
        for r in records
    }
    options = list(labels.keys())

    col1, col2 = st.columns(2)
    with col1:
        base_label = st.selectbox("Baseline (good)", options, index=0)
    with col2:
        cand_label = st.selectbox("Candidate (new)", options, index=min(1, len(options) - 1))

    base_id = labels[base_label]
    cand_id = labels[cand_label]

    if base_id == cand_id:
        st.info("Pick two different runs to compare.")
        return

    baseline = store.get_run(base_id)
    candidate = store.get_run(cand_id)
    if baseline is None or candidate is None:
        st.error("Run not found — try refreshing.")
        return

    if baseline.kind != candidate.kind:
        st.warning(
            f"Cannot compare different kinds: {baseline.kind} vs {candidate.kind}."
        )
        return

    all_metrics = sorted(set(baseline.metrics) | set(candidate.metrics))
    rows = []
    for name in all_metrics:
        b = baseline.metrics.get(name, 0.0)
        c = candidate.metrics.get(name, 0.0)
        delta = c - b
        pct = (delta / b * 100) if b != 0 else 0.0
        rows.append(
            {
                "metric": name,
                "baseline": round(b, 4),
                "candidate": round(c, 4),
                "delta": round(delta, 4),
                "pct": f"{pct:+.1f}%" if b != 0 else "—",
            }
        )

    df = pd.DataFrame(rows)

    # Summary cards
    regressions = df[df["delta"] < -0.001]
    improvements = df[df["delta"] > 0.001]

    c1, c2, c3 = st.columns(3)
    c1.metric("Regressions", len(regressions))
    c2.metric("Improvements", len(improvements))
    c3.metric("Unchanged", len(df) - len(regressions) - len(improvements))

    # Full delta table with color
    st.markdown("### Metric deltas")

    def _color_row(row):
        if row["delta"] < -0.001:
            return ["background-color: #ffe5e5"] * len(row)
        if row["delta"] > 0.001:
            return ["background-color: #e5ffe5"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_color_row, axis=1).format(
        {"baseline": "{:.4f}", "candidate": "{:.4f}", "delta": "{:+.4f}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if len(regressions) > 0:
        st.error(
            f"{len(regressions)} metric(s) regressed. "
            f"Worst: **{regressions.sort_values('delta').iloc[0]['metric']}** "
            f"({regressions.sort_values('delta').iloc[0]['delta']:+.4f})"
        )
    elif len(improvements) > 0:
        st.success(
            f"All changes positive or neutral. "
            f"{len(improvements)} metric(s) improved."
        )
    else:
        st.info("No meaningful change between these runs.")

def _render_gate_view(store: SQLiteRunStore) -> None:
    st.subheader("Gate")

    records = store.list_runs(limit=100)
    if len(records) < 2:
        st.info("Need at least two runs to gate. Start more with `POST /eval/run`.")
        return

    labels = {
        f"{r.id[:12]} | {r.kind} | {r.retriever_name} | {r.created_at}": r.id
        for r in records
    }
    options = list(labels.keys())

    col1, col2 = st.columns(2)
    with col1:
        base_label = st.selectbox("Baseline (good)", options, index=0, key="gate_base")
    with col2:
        cand_label = st.selectbox(
            "Candidate (new)", options, index=min(1, len(options) - 1), key="gate_cand"
        )

    base_id = labels[base_label]
    cand_id = labels[cand_label]

    if base_id == cand_id:
        st.info("Pick two different runs to gate.")
        return

    baseline = store.get_run(base_id)
    candidate = store.get_run(cand_id)
    if baseline is None or candidate is None:
        st.error("Run not found — try refreshing.")
        return

    if baseline.kind != candidate.kind:
        st.warning(
            f"Cannot gate different kinds: {baseline.kind} vs {candidate.kind}."
        )
        return

    # Load thresholds for this kind
    try:
        thresholds = load_thresholds(baseline.kind, DEFAULT_THRESHOLDS_PATH)
    except ThresholdsConfigError as e:
        st.error(f"Thresholds error: {e}")
        return

    # Run the gate
    result = evaluate_gate(
        baseline_metrics=baseline.metrics,
        candidate_metrics=candidate.metrics,
        thresholds=thresholds,
    )

    # Verdict card
    if result.passed:
        st.success("## PASS")
    else:
        st.error("## FAIL")

    st.caption(
        f"Thresholds: `{DEFAULT_THRESHOLDS_PATH}`  •  kind: `{baseline.kind}`"
    )

    # Regressions
    if result.regressions:
        st.markdown("### Regressions")
        reg_rows = [
            {
                "metric": r.metric,
                "baseline": round(r.baseline, 4),
                "candidate": round(r.candidate, 4),
                "delta": round(r.delta, 4),
                "threshold": round(r.threshold, 4),
                "gated": "yes" if r.threshold > 0 else "no",
            }
            for r in result.regressions
        ]
        st.dataframe(
            pd.DataFrame(reg_rows), use_container_width=True, hide_index=True
        )

    # Improvements
    if result.improvements:
        st.markdown("### Improvements")
        imp_rows = [
            {
                "metric": i.metric,
                "baseline": round(i.baseline, 4),
                "candidate": round(i.candidate, 4),
                "delta": round(i.delta, 4),
            }
            for i in result.improvements
        ]
        st.dataframe(
            pd.DataFrame(imp_rows), use_container_width=True, hide_index=True
        )

    # Unchanged
    if result.unchanged:
        st.markdown("### Unchanged (within threshold)")
        st.write(", ".join(f"`{m}`" for m in result.unchanged))

# ---------- sidebar ----------

st.sidebar.header("RAGGate AI")
db_path = st.sidebar.text_input("Database path", value=settings.db_path)
st.sidebar.caption("Same DB the API writes to. Leave default unless you know why.")

if not Path(db_path).exists():
    st.warning(f"No database at `{db_path}`. Start the API and run an eval first.")
    st.stop()

store = get_store(db_path)
st.sidebar.button("Refresh")  # clicking reruns the script

# ---------- views ----------

st.title("RAGGate AI")
st.caption("Automated evaluation harness and regression gate for RAG pipelines.")

view = st.sidebar.radio("View", ["Runs", "Compare", "Gate"])

if view == "Runs":
    _render_runs_view(store)
elif view == "Compare":
    _render_compare_view(store)
elif view == "Gate":
    _render_gate_view(store)