"""FastAPI entrypoint for RAGGate AI."""

from fastapi import FastAPI

from raggate import __version__
from raggate.config import settings

app = FastAPI(
    title="RAGGate AI",
    description="Automated evaluation harness and regression gate for RAG pipelines.",
    version=__version__,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "env": settings.env,
    }