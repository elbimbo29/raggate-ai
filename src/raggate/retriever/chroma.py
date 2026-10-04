"""Chroma-backed retriever.

Embeds each corpus chunk once at construction time using a
sentence-transformers model, stores the vectors in an ephemeral Chroma
collection, and retrieves by cosine similarity.

Ephemeral (in-memory) is deliberate — the corpus is tiny, and we don't
want a persistent store on disk just for evaluation.
"""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from raggate.retriever.corpus import Chunk

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class ChromaRetriever:
    """Retrieve chunks by embedding similarity."""

    def __init__(
        self,
        chunks: list[Chunk],
        name: str = "chroma-minilm",
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self._name = name
        self._chunks = chunks
        self._model_name = model_name

        self._client = chromadb.Client()  # ephemeral, in-memory
        self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self._collection = self._client.create_collection(
            name="acmedb",
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"section": c.section} for c in chunks],
        )

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, question: str, k: int = 5) -> list[str]:
        n = min(k, len(self._chunks))
        result = self._collection.query(query_texts=[question], n_results=n)
        # Chroma returns ids as list[list[str]]; we want the inner list.
        ids = result.get("ids") or [[]]
        return list(ids[0])