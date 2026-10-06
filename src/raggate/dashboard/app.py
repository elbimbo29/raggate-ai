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

view = st.sidebar.radio("View", ["Runs"])

if view == "Runs":
    _render_runs_view(store)