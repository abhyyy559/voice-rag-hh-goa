"""
Retrieval layer.

Design choice: everything here runs in-process (BM25 + TF-IDF cosine, no
network hop to a hosted vector DB). A round trip to Pinecone/Weaviate/etc.
typically costs 30-150ms on its own before you've retrieved anything, which
eats most of the 200ms budget before generation even starts. Local
BM25+TF-IDF hybrid retrieval instead runs in low single-digit milliseconds
for corpora of this size, which is what makes the latency target achievable.

The coding agent can swap the TF-IDF vectorizer for real dense embeddings
(sentence-transformers / an API embedding model) once it has network access —
the VectorIndex interface below doesn't change either way, only the
_embed() implementation would.
"""
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z\u0900-\u097F]+", text.lower())  # incl. Devanagari range for Hindi


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    bm25_score: float
    tfidf_score: float


class VectorIndex:
    """Hybrid lexical (BM25) + vector-space (TF-IDF cosine) index.
    Swap-in point for real embeddings later — see module docstring."""

    def __init__(self, chunks: List[Chunk], alpha: float = 0.5):
        self.chunks = chunks
        self.alpha = alpha
        self._build()

    def _build(self):
        texts = [c.text for c in self.chunks]
        self._tokenized = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self._tokenized) if texts else None
        self.vectorizer = TfidfVectorizer(tokenizer=_tokenize, lowercase=False) if texts else None
        self.tfidf_matrix = self.vectorizer.fit_transform(texts) if texts else None

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        mx = float(scores.max()) or 1e-9
        mn = float(scores.min())
        rng = (mx - mn) or 1e-9
        return (scores - mn) / rng

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.chunks:
            return []
        q_tokens = _tokenize(query)
        bm25_raw = np.asarray(self.bm25.get_scores(q_tokens) if q_tokens else [0.0] * len(self.chunks))
        q_vec = self.vectorizer.transform([query])
        tfidf_raw = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Vectorized combined scoring + top-k (avoids a Python loop over all
        # docs, which dominated latency at 20k+ chunks).
        combined = (self.alpha * self._normalize(tfidf_raw)
                    + (1 - self.alpha) * self._normalize(bm25_raw))
        order = np.argsort(-combined)[:top_k]
        return [
            RetrievalResult(
                chunk=self.chunks[i], score=float(combined[i]),
                bm25_score=float(bm25_raw[i]), tfidf_score=float(tfidf_raw[i]),
            )
            for i in order
        ]
