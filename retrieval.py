"""Embedding + retrieval helpers.

Two jobs live here:

1. `Embedder` wraps a local sentence-transformers model and produces
   L2-normalized vectors. The same instance is reused for the RAG retrieval
   query *and* for the round-to-round convergence signal, so both use an
   identical embedding space.

2. `ReferenceLibrary` holds one run's reference summaries (written fresh per
   clip -- see agents.generate_references), pre-embeds them once, and answers
   `top_k_similar(query_text, k)` by brute-force cosine similarity. At a few
   dozen entries a real vector DB would be overkill.

`cosine_distance` measures round-to-round drift:  d_t = 1 - cos(e_t, e_{t-1}).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

import config

try:
    # Use the OS trust store for TLS so model downloads work behind a
    # corporate proxy that does MITM certificate inspection.
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - best effort; falls back to certifi
    pass


@lru_cache(maxsize=2)
def _load_model(name: str):
    """Load and cache a sentence-transformers model by name."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class Embedder:
    """Local, free sentence embedder producing unit-norm vectors."""

    def __init__(self, model_name: str = config.EMBED_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def encode(self, text: str) -> np.ndarray:
        """Return the L2-normalized embedding of `text` as a float32 vector."""
        model = self._ensure()
        vec = model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (safe for non-normalized input)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cos(a, b), clamped to the valid range [0, 2]."""
    d = 1.0 - cosine_similarity(a, b)
    return float(min(2.0, max(0.0, d)))


@dataclass
class RetrievedExample:
    id: str
    category: str
    text: str
    similarity: float

    def as_dict(self, include_text: bool = True) -> dict[str, Any]:
        d = {
            "id": self.id,
            "category": self.category,
            "similarity": round(self.similarity, 4),
        }
        if include_text:
            d["text"] = self.text
        return d


class ReferenceLibrary:
    """A pre-embedded corpus of reference-quality video summaries."""

    def __init__(self, entries: list[dict[str, Any]], embedder: Embedder) -> None:
        if not entries:
            raise ValueError("reference library is empty")
        for e in entries:
            missing = {"id", "category", "text"} - set(e)
            if missing:
                raise ValueError(f"reference entry {e!r} missing keys: {missing}")
        self.entries = entries
        self.embedder = embedder
        self._matrix = np.vstack([embedder.encode(e["text"]) for e in entries])

    def __len__(self) -> int:
        return len(self.entries)

    def top_k_similar(self, query_text: str, k: int = config.TOP_K) -> list[RetrievedExample]:
        """Return the k most cosine-similar reference summaries, best first."""
        q = self.embedder.encode(query_text)
        # rows of _matrix are already unit-norm, q is unit-norm -> dot == cosine
        sims = self._matrix @ q
        k = max(1, min(k, len(self.entries)))
        order = np.argsort(-sims)[:k]
        return [
            RetrievedExample(
                id=self.entries[i]["id"],
                category=self.entries[i]["category"],
                text=self.entries[i]["text"],
                similarity=float(sims[i]),
            )
            for i in order
        ]
