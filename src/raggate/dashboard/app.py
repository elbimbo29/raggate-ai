"""RAGGate AI — Streamlit dashboard.

Reads from the same SQLite store the API writes to. No LLM calls, no
cost. Three views land over the next few steps: Runs, Compare, Gate.

Run with:
    make dashboard
or:
    uv run streamlit run src/raggate/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="RAGGate AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("RAGGate AI")
st.caption("Automated evaluation harness and regression gate for RAG pipelines.")

st.info("Dashboard under construction — views land in Steps 6.2–6.4.")