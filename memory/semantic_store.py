"""
SemanticStore — similarity search over past events, backed by Pinecone.

Text is embedded locally with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
and upserted to a Pinecone serverless index. Queries embed the query text the
same way and return the nearest stored vectors by cosine similarity — letting the
system find semantically related precedents regardless of exact wording.

Pairs with EpisodicStore: the same event can live in both, sharing an id.
Episodic answers exact filters; semantic answers "have we seen anything like this?".

Design: lazy model + client (nothing loads on import), fire-and-forget resilient
(store returns an error dict, query returns [] — never crashes an agent).
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from config import settings

_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singleton — the ~90 MB model is loaded once and shared.
_model = None


def _get_model():
    """Lazily load and cache the embedding model. Returns None if unavailable."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    except Exception:
        _model = None
    return _model


def _embed(text: str) -> Optional[list[float]]:
    """Embed one string into a 384-dim vector, or None on failure."""
    model = _get_model()
    if model is None:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception:
        return None


class SemanticStore:
    def __init__(self) -> None:
        self._index = None  # lazy Pinecone index handle

    # ── Connection ─────────────────────────────────────────────────────────────

    def _get_index(self):
        """Lazily create the Pinecone index handle. Returns None if unconfigured."""
        if self._index is not None:
            return self._index
        if not settings.PINECONE_API_KEY:
            return None
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self._index = pc.Index(settings.PINECONE_INDEX)
        except Exception:
            self._index = None
        return self._index

    # ── Write ──────────────────────────────────────────────────────────────────

    def store(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> dict:
        """
        Embed text and upsert it as a vector. Never raises.
        Returns {"status": "ok", "id": ...} or {"status": "error", "error": ...}.
        """
        index = self._get_index()
        if index is None:
            return {"status": "error", "error": "pinecone unavailable"}

        vector = _embed(text)
        if vector is None:
            return {"status": "error", "error": "embedding model unavailable"}

        vec_id = id or str(uuid.uuid4())
        # Pinecone metadata must be flat scalars/strings; keep the raw text for recall.
        meta = {"text": text, **(metadata or {})}
        try:
            index.upsert(vectors=[{"id": vec_id, "values": vector, "metadata": meta}])
            return {"status": "ok", "id": vec_id}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ── Read ───────────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int = 3,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return the top_k most similar stored vectors, highest score first.
        Each item: {"id", "score", "metadata"}. Returns [] on error.
        """
        index = self._get_index()
        if index is None:
            return []

        vector = _embed(text)
        if vector is None:
            return []

        try:
            resp = index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter,
            )
            matches = resp.get("matches", []) if isinstance(resp, dict) else resp.matches
            results = [
                {
                    "id": m["id"] if isinstance(m, dict) else m.id,
                    "score": m["score"] if isinstance(m, dict) else m.score,
                    "metadata": (m.get("metadata") if isinstance(m, dict) else m.metadata) or {},
                }
                for m in matches
            ]
            # Guarantee highest-similarity-first regardless of backend ordering.
            results.sort(key=lambda r: r["score"], reverse=True)
            return results
        except Exception:
            return []

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete(self, id: str) -> dict:
        """Remove one vector by id. Never raises."""
        index = self._get_index()
        if index is None:
            return {"status": "error", "error": "pinecone unavailable"}
        try:
            index.delete(ids=[id])
            return {"status": "ok", "id": id}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
