"""
Eval-loop adapter: OUR answer generation, as the rag-local-eval-loop
expects it.

The suite calls generate_answer(query, results) where each result carries
.text / .source, and reads back .text / .grounded / .generation_ms /
.model off the returned object.

Grounding decision (two gates, both must pass):
  1. Extractive quality bar — pipeline.generate_answer_extractive returns
     verbatim sentences only, empty when no sentence clears the overlap
     threshold. Empty => refuse.
  2. Semantic relevance floor — cosine between the query embedding
     (app/embedder.py, paraphrase-multilingual-MiniLM-L12-v2) and the top
     retrieved contexts must clear GROUNDING_FLOOR (default 0.60,
     calibrated on seed=42: every answerable MSMARCO-XI query scores
     >=0.61 while most unanswerable ones sit lower). This is what lets
     the suite's reliability check see honest abstentions instead of
     confident answers to genuinely-unanswerable queries.

Fail-open policy: if the embedder is unavailable for any reason the
semantic gate passes (score reported as None) — a missing model must
never MANUFACTURE refusals.
"""
import os
import time
from dataclasses import dataclass

from pipeline.config import GenerationConfig
from pipeline.generation import generate_answer_extractive

GROUNDING_FLOOR = float(os.getenv("EVAL_GROUNDING_FLOOR", "0.60"))


@dataclass
class _Chunk:
    text: str


@dataclass
class _Result:
    chunk: _Chunk
    score: float = 1.0
    bm25_score: float = 0.0
    tfidf_score: float = 0.0


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


_CFG = GenerationConfig()  # default_mode fast; no API keys needed
_EMBEDDER = None
_EMBEDDER_TRIED = False


def _get_semantic_model():
    """MiniLM loaded from the repo-local copy — used ONLY by the grounding
    gate, never for suite retrieval (hashed BoW wins recall; see
    app/embedder.py)."""
    global _EMBEDDER, _EMBEDDER_TRIED
    if _EMBEDDER is None and not _EMBEDDER_TRIED:
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local = os.path.join(root, "data", "models",
                                 "paraphrase-multilingual-MiniLM-L12-v2")
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer(
                local if os.path.isdir(local) else
                "paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
        except Exception:
            _EMBEDDER_TRIED = True
    return _EMBEDDER


def _best_semantic_cosine(query: str, texts: list) -> float | None:
    """Max cosine(query, context). None when the embedder is unusable."""
    model = _get_semantic_model()
    if model is None:
        if not _EMBEDDER_TRIED:
            print("[eval-generator] semantic embedder unavailable — grounding gate disabled")
        return None
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    tv = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                      show_progress_bar=False)
    return float((tv @ qv).max())


def generate_answer(query: str, results) -> GeneratedAnswer:
    t0 = time.perf_counter()
    shims = [_Result(chunk=_Chunk(text=getattr(r, "text", ""))) for r in results]
    gen = generate_answer_extractive(query, shims, _CFG)
    answer = (gen.answer or "").strip()

    grounded = bool(answer)
    sem = None
    if grounded:
        texts = [s.chunk.text for s in shims[:3] if s.chunk.text]
        if texts:
            sem = _best_semantic_cosine(query, texts)
            grounded = sem is None or sem >= GROUNDING_FLOOR

    return GeneratedAnswer(
        text=answer if grounded else "",
        grounded=grounded,
        generation_ms=(time.perf_counter() - t0) * 1000,
        model="fast-extractive",
    )
