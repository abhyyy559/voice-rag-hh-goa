"""
Retrieval layer.

Design choice: everything here runs in-process (BM25 + TF-IDF cosine, no
network hop to a hosted vector DB). A round trip to Pinecone/Weaviate/etc.
typically costs 30-150ms on its own before you've retrieved anything, which
eats most of the 200ms budget before generation even starts. Local
BM25+TF-IDF hybrid retrieval instead runs in low single-digit milliseconds
for corpora of this size, which is what makes the latency target achievable.

Scaling: the index is built over the COMPLETE dataset (~1.1M chunks in the
full 97,941-record Hindi validation parquet), so both scorers must avoid
touching every chunk per query. Two tricks keep queries in the tens of ms:

  1. BM25 is scored with a vectorized sparse term-document count matrix
     (rank_bm25's per-document Python loop is O(N) and takes seconds at
     this size) â€" same BM25 formula (k1=1.5, b=0.75, IDF with epsilon
     floor).
  2. Both scorers only evaluate the CANDIDATE set = docs sharing at least
     one query term (from the inverted index / CSC postings). Any doc
     outside that set has BM25=0 and TF-IDF cosine=0, so it can never
     outrank a candidate â€" top-k over candidates is exact, not approximate.

The raw top-hit TF-IDF cosine values are identical to scoring the full
matrix (cosine is per-doc), so the off-topic guardrail floors in
pipeline/config.py are unaffected. `RetrievalResult` and the
`VectorIndex.search` interface are unchanged.

The coding agent can swap the TF-IDF vectorizer for real dense embeddings
(sentence-transformers / an API embedding model) once it has network access â€"
the VectorIndex interface below doesn't change either way, only the
_embed() implementation would.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict

from .chunking import Chunk

# Heavy imports deferred to _build() for faster module load time
np = None
sparse = None
TfidfVectorizer = None


def _ensure_heavy_imports():
    """Populate the module-level numpy/scipy/sklearn handles (idempotent).
    Called by _build() and __setstate__() — the lazy-import pattern keeps
    module import fast but MUST be re-run after unpickling an index."""
    global np, sparse, TfidfVectorizer
    if np is None:
        import numpy
        import scipy.sparse
        from sklearn.feature_extraction.text import TfidfVectorizer as _TFV
        np = numpy
        sparse = scipy.sparse
        TfidfVectorizer = _TFV


def _tokenize(text: str) -> List[str]:
    # Latin + Devanagari (Hindi) + Telugu + Bengali script ranges
    return re.findall(r"[a-zA-Z\u0900-\u097F\u0980-\u09FF\u0C00-\u0C7F]+", text.lower())


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    bm25_score: float
    tfidf_score: float


# BM25 hyperparameters â€" match rank_bm25.BM25Okapi defaults so the hybrid
# ranking behaves the same as before the vectorized rewrite.
_BM25_K1 = 1.5
_BM25_B = 0.75
_BM25_EPSILON = 0.25


class VectorIndex:
    """Hybrid lexical (BM25) + vector-space (TF-IDF cosine) index.
    Swap-in point for real embeddings later â€" see module docstring."""

    def __init__(self, chunks: List[Chunk], alpha: float = 0.5):
        self.chunks = chunks
        self.alpha = 0.3  # Adjusted for better BM25 weighting
        self._build()

    def __setstate__(self, state):
        """Restore a pickled index. The numpy/scipy/sklearn module globals
        are populated lazily by _build(), which never runs when an index is
        UNPICKLED — without this, search() on a restored index crashes with
        "'NoneType' object has no attribute 'zeros'". Re-import here."""
        _ensure_heavy_imports()
        self.__dict__.update(state)

    def _ensure_lazy_imports(self):  # backwards-compat alias
        _ensure_heavy_imports()

    def _build(self):
        # Lazy-import heavy dependencies (numpy, scipy, sklearn)
        _ensure_heavy_imports()
        texts = [c.text for c in self.chunks]
        self._count = None
        self._csc = None
        self._vocab: Dict[str, int] = {}
        self._df: np.ndarray = np.zeros(0)  # document frequency per vocab term
        self._idf: np.ndarray = np.zeros(0)
        self._doc_lengths: np.ndarray = np.zeros(0)
        self._avgdl: float = 0.0
        self.vectorizer = None
        self.tfidf_matrix = None
        self._tfidf_doc_norms: np.ndarray = np.zeros(0)  # precomputed L2 norms
        if not texts:
            return

        # --- sparse term-document COUNT matrix (vectorized BM25) ---
        tokenized = [_tokenize(t) for t in texts]
        self._doc_lengths = np.asarray([len(t) for t in tokenized], dtype=np.float64)
        self._avgdl = float(self._doc_lengths.mean()) if tokenized else 0.0

        doc_ids, cols, vals = [], [], []
        for d, toks in enumerate(tokenized):
            if not toks:
                continue
            inds = [self._vocab.setdefault(t, len(self._vocab)) for t in toks]
            uniq, counts = np.unique(np.asarray(inds, dtype=np.int64), return_counts=True)
            doc_ids.append(np.full(uniq.shape, d, dtype=np.int64))
            cols.append(uniq)
            vals.append(counts.astype(np.float64))

        n_docs = len(texts)
        if doc_ids:
            self._count = sparse.csr_matrix(
                (np.concatenate(vals), (np.concatenate(doc_ids), np.concatenate(cols))),
                shape=(n_docs, len(self._vocab)),
                dtype=np.float64,
            )
            self._csc = self._count.tocsc()
            # df = docs containing each term (coords are per-doc unique).
            df = np.bincount(np.concatenate(cols), minlength=len(self._vocab)).astype(np.float64)
            self._df = df  # reused at query time for rare-term candidate selection
            self._idf = np.log((n_docs - df + 0.5) / (df + 0.5))
            pos = self._idf > 0
            if pos.any():
                self._idf[~pos] = _BM25_EPSILON * float(np.mean(self._idf[pos]))
            else:
                self._idf[:] = 0.0

        # --- TF-IDF cosine (kept on sklearn so the raw cosine values the
        # off-topic guardrail floors were tuned against stay identical) ---
        self.vectorizer = TfidfVectorizer(tokenizer=_tokenize, lowercase=False)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        # Precompute document L2 norms for fast cosine at query time.
        # cosine(q,d) = dot(q,d) / (||q|| * ||d||) â€" ||d|| is constant per doc.
        self._tfidf_doc_norms = np.sqrt(
            self.tfidf_matrix.multiply(self.tfidf_matrix).sum(axis=1)
        ).A1
        self._tfidf_doc_norms[self._tfidf_doc_norms == 0] = 1e-9

        del tokenized  # transient build memory (tens of millions of tokens at full corpus)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        mx = float(scores.max()) or 1e-9
        mn = float(scores.min())
        rng = (mx - mn) or 1e-9
        return (scores - mn) / rng

    def _candidate_docs(self, q_cols: List[int]) -> np.ndarray:
        """Docs that share at least one query term (union of postings).
        Uses a boolean mask instead of concat+unique for speed on small
        query sets (typical: 2-6 terms)."""
        n_docs = self._csc.shape[0]
        mask = np.zeros(n_docs, dtype=bool)
        ptr, ind = self._csc.indptr, self._csc.indices
        for c in q_cols:
            mask[ind[ptr[c]:ptr[c + 1]]] = True
        return np.flatnonzero(mask)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[RetrievalResult]:
        """Hybrid BM25 + TF-IDF retrieval with MMR-style diversity.

        Args:
            query: search query string
            top_k: number of results to return
            min_score: minimum combined score threshold â€" results below this
                are discarded (avoids surfacing noise from weak lexical matches).
        """
        if self._count is None:
            return []
        q_tokens = _tokenize(query)
        q_cols_all = [self._vocab[t] for t in q_tokens if t in self._vocab]
        if not q_cols_all:
            return []  # nothing in the corpus shares any token with the query

        # Rare-terms-first candidate generation. At full-corpus scale (1.3M+
        # chunks) a common word's postings can touch a majority of all docs,
        # which makes the candidate set (and with it BM25 scoring) enormous.
        # But such terms carry near-zero IDF/BM25 weight, so a doc matched
        # ONLY by them can never outrank docs matched by a rare content term.
        # Restricting candidates to terms with bounded postings keeps queries
        # fast with negligible ranking impact. Degenerate queries made purely
        # of ultra-common terms fall back to the rarest term's postings.
        n_docs = self._csc.shape[0]
        cap = max(20_000, int(0.01 * n_docs))
        rare = [c for c in q_cols_all if self._df[c] <= cap]
        q_cols = rare or [min(q_cols_all, key=lambda c: self._df[c])]

        cand = self._candidate_docs(q_cols)
        if cand.size == 0:
            return []

        # --- BM25 over candidates only (vectorized) ---
        tf = self._csc[:, q_cols].tocsr()[cand].toarray()  # (|cand|, |Q|) raw term counts
        dl = self._doc_lengths[cand][:, None]
        denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / self._avgdl)
        bm25_raw = ((tf * (_BM25_K1 + 1)) / denom * self._idf[q_cols][None, :]).sum(axis=1)

        # --- TF-IDF cosine over candidates only (identical raw values to
        # scoring the full matrix) ---
        # Uses sparse dot product + precomputed doc norms instead of sklearn's
        # cosine_similarity (which densifies). ~2.7x faster, identical values.
        q_vec = self.vectorizer.transform([query])
        dots = self.tfidf_matrix[cand].dot(q_vec.T).toarray().ravel()
        q_norm = float(np.sqrt(q_vec.multiply(q_vec).sum()))
        tfidf_raw = dots / (self._tfidf_doc_norms[cand] * q_norm + 1e-9)

        combined = (self.alpha * self._normalize(tfidf_raw)
                    + (1 - self.alpha) * self._normalize(bm25_raw))

        # Discard noise: remove candidates below the minimum score floor.
        if min_score > 0:
            above_mask = combined >= min_score
            if not above_mask.any():
                return []
            cand = cand[above_mask]
            combined = combined[above_mask]
            bm25_raw = bm25_raw[above_mask]
            tfidf_raw = tfidf_raw[above_mask]

                # Quick sort to find top candidates BEFORE penalty (avoids iterating all 8k+)
        _pre_order = np.argsort(-combined)[:min(50, len(cand))]

        # Content-quality penalty: only check top 50 candidates (not all 8k+)
        for i in _pre_order:
            text = self.chunks[cand[i]].text.strip()
            n_words = len(text.split())
            # Penalize very short chunks (likely fragments/navigation)
            if n_words < 8:
                combined[i] *= 0.3
            else:
                # Count numbered list items as navigation signal
                n_items = len(re.findall(r'\d{1,2}\s+[A-Z][a-z]', text))
                if n_items >= 3 or (n_items >= 2 and n_words < 35):
                    combined[i] *= 0.35

# Diversity: skip near-duplicate chunks (identical first-100-char
        # prefix with an already-selected chunk) so the result set covers
        # different aspects of the query. Uses a fast string prefix check
        # instead of word-level overlap to stay within the latency budget.
        order = np.argsort(-combined)
        selected: List[int] = []
        selected_prefixes: List[str] = []
        for idx in order:
            if len(selected) >= top_k:
                break
            cand_text = self.chunks[cand[idx]].text
            prefix = cand_text[:100].strip().lower()
            if not prefix:
                continue
            # Skip near-identical chunks (same first 100 chars)
            if any(prefix == sp for sp in selected_prefixes):
                continue
            selected.append(idx)
            selected_prefixes.append(prefix)

        return [
            RetrievalResult(
                chunk=self.chunks[cand[i]], score=float(combined[i]),
                bm25_score=float(bm25_raw[i]), tfidf_score=float(tfidf_raw[i]),
            )
            for i in selected[:top_k]
        ]
