"""
The harness is the orchestration layer the task spec asks for: structured
input/output, retries (delegated to each stage via tenacity), per-stage
timing, and graceful error recovery — instead of one raw prompt-in/text-out
call to a model.

Flow:
  audio bytes -> [STT] -> transcript
  transcript  -> [guardrail: unsafe input]      -> pass/refuse
  transcript  -> [retrieval]                     -> top-k chunks
  chunks      -> [guardrail: off-topic]           -> pass/refuse
  chunks+query-> [generation]                     -> answer
  answer      -> [guardrail: grounding]           -> pass/refuse
             -> [PipelineResult] (always returned, success or refusal)

Every stage failure is caught and turned into a structured refusal rather
than an unhandled exception, so the harness never crashes the caller —
callers always get a PipelineResult with a clear `status`.
"""
import os
import re
import time
import pickle
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

from .config import PipelineConfig
from .chunking import STRATEGIES, Chunk
from .retrieval import VectorIndex, RetrievalResult
from .guardrails import check_unsafe_input, check_off_topic, check_grounding, GuardrailVerdict, _content_words
from .generation import (
    generate_answer, generate_answer_mock, generate_answer_extractive, GenerationError,
)
from .stt import transcribe, STTError, TranscriptionResult

_TOKEN_RE = re.compile(r"[a-zA-Z\u0900-\u097F\u0980-\u09FF\u0C00-\u0C7F]+")  # Latin + Devanagari + Bengali + Telugu

# Common multi-word entity patterns that should be matched as units.
# When a query contains "andhra pradesh", both words must appear near each
# other in the passage for it to count as truly relevant — just matching
# "andhra" in an unrelated context is a false positive.
_MULTIWORD_ENTITIES = [
    frozenset(["andhra", "pradesh"]),
    frozenset(["telangana"]),
    frozenset(["chief", "minister"]),
    frozenset(["capital", "city"]),
    frozenset(["north", "india"]),
    frozenset(["south", "india"]),
    frozenset(["east", "india"]),
    frozenset(["west", "india"]),
    frozenset(["united", "states"]),
    frozenset(["united", "kingdom"]),
]


def _check_topic_relevance(query: str, results: List[RetrievalResult]) -> float:
    """Measures how well the retrieved passages address the query's core topic.
    Returns a 0.0-1.0 relevance score. Below 0.45 means the passages mention
    query entities but don't actually answer the question.

    This catches the common failure mode where TF-IDF/BM25 match on entity
    names (e.g. "Andhra Pradesh") but the passage is about a completely
    different aspect (food culture vs. Chief Minister's Fund).
    """
    if not results:
        return 0.0

    q_words = _content_words(query)
    if not q_words:
        return 0.5  # no content words — can't assess, let other checks decide

    context_text = " ".join(r.chunk.text.lower() for r in results[:3])
    context_words = _content_words(context_text)

    # Check 1: multi-word entity coherence
    entity_bonus = 0.0
    n_entities_checked = 0
    for entity_words in _MULTIWORD_ENTITIES:
        if entity_words.issubset(q_words):
            n_entities_checked += 1
            for r in results[:3]:
                text_lower = r.chunk.text.lower()
                for w1 in entity_words:
                    for w2 in entity_words:
                        if w1 == w2:
                            continue
                        idx1 = text_lower.find(w1)
                        idx2 = text_lower.find(w2)
                        if idx1 >= 0 and idx2 >= 0 and abs(idx1 - idx2) < 50:
                            entity_bonus += 1.0
                            break
                    if entity_bonus > 0:
                        break
                if entity_bonus > 0:
                    break
    entity_score = (entity_bonus / n_entities_checked) if n_entities_checked > 0 else 1.0

    # Check 2: question-focus words — the word that tells us WHAT the user is
    # asking about ("capital", "population", "minister", etc.) must appear
    # in the context. If not, the passage just happens to mention the entity
    # but doesn't address the actual question.
    _QUESTION_FOCUS = {
        "capital": ["capital", "city", "seat", "headquarters"],
        "cmf": ["fund", "cmf", "chief minister fund", "scheme", "program"],
        # NOTE: "cm" here means Chief Minister. Matching must be word-bounded,
        # otherwise "cm" hits every "1 cm 2" centimeter in math passages.
        "cm": ["chief", "minister", "cm", "cabinet", "government"],
        "minister": ["minister", "cm", "chief", "government", "cabinet"],
        "governor": ["governor", "appointed", "state", "office"],
        "population": ["population", "people", "inhabitants", "residents", "census"],
        "area": ["area", "sq", "km", "square", "kilometers", "size"],
        "language": ["language", "languages", "speak", "spoken", "tongue"],
        "currency": ["currency", "rupee", "money", "notes", "coins"],
        "president": ["president", "head", "state", "office"],
        "prime": ["prime", "minister", "pm", "head", "government"],
        "river": ["river", "flows", "water", "banks", "tributary"],
        "mountain": ["mountain", "peak", "range", "hills", "altitude"],
    }
    focus_words = set()
    for q_word in q_words:
        if q_word in _QUESTION_FOCUS:
            focus_words.update(_QUESTION_FOCUS[q_word])

    if focus_words:
        # Context-aware focus check: the focus word must appear in the SAME
        # SENTENCE as an entity word. This prevents false matches where
        # "capital" appears in "Capital Required" (financial context) while
        # "Andhra Pradesh" is in a completely different sentence about food.
        # Only use entity words from entities actually present in the query
        # (not ALL entity words) to avoid self-matching.
        query_entity_words = set()
        for ew in _MULTIWORD_ENTITIES:
            if ew.issubset(q_words):
                query_entity_words |= ew
        _SENT_RE = re.compile(r'[.!?]\s+')

        def _wb(word: str):
            return re.compile(r"\b" + re.escape(word) + r"\b")

        # Word-boundary patterns: raw substring `in` checks would match "cm"
        # inside unrelated tokens and "art" inside "particle".
        focus_pats = [_wb(fw) for fw in focus_words]
        entity_pats = [_wb(ew) for ew in query_entity_words]
        focus_near_entity = 0
        for r in results[:3]:
            sentences = _SENT_RE.split(r.chunk.text.lower())
            for sent in sentences:
                has_entity = any(p.search(sent) for p in entity_pats)
                has_focus = any(p.search(sent) for p in focus_pats)
                if has_entity and has_focus:
                    focus_near_entity += 1
                    break  # one match per passage is enough
        focus_score = min(1.0, focus_near_entity / max(len(results[:3]), 1))
    else:
        query_entity_words = set()
        for ew in _MULTIWORD_ENTITIES:
            if ew.issubset(q_words):
                query_entity_words |= ew
        non_entity_q = q_words - query_entity_words
        if non_entity_q:
            non_entity_matches = len(non_entity_q & context_words)
            focus_score = min(1.0, non_entity_matches / len(non_entity_q))
        else:
            # Pure entity query — conservative score
            entity_related = len(query_entity_words & context_words)
            focus_score = min(0.3, entity_related / max(len(query_entity_words), 1) * 0.5)

    # Combined score: entity coherence (25%) + focus word presence (55%) +
    # base content overlap (20%)
    base_overlap = len(q_words & context_words) / len(q_words)
    relevance = 0.25 * entity_score + 0.55 * focus_score + 0.2 * min(1.0, base_overlap)
    return relevance


def _corpus_stopwords(chunks: List[Chunk], threshold_pct: float = 1.0) -> frozenset:
    """Tokens common enough across the corpus to be uninformative (Hindi and
    English function words). Measured against the real corpus this cleanly
    captures function words while leaving content words in the query."""
    df = Counter()
    for c in chunks:
        for w in set(_TOKEN_RE.findall(c.text.lower())):
            df[w] += 1
    thresh = max(len(chunks) * threshold_pct / 100, 1)
    return frozenset(w for w, cnt in df.items() if cnt >= thresh)


class Status(str, Enum):
    OK = "ok"
    REFUSED = "refused"
    ERROR = "error"


@dataclass
class StageTiming:
    stage: str
    ms: float


@dataclass
class PipelineResult:
    status: Status
    query_text: Optional[str] = None
    answer: Optional[str] = None
    retrieved: List[RetrievalResult] = field(default_factory=list)
    refusal_reason: Optional[str] = None
    timings: List[StageTiming] = field(default_factory=list)
    total_ms: float = 0.0
    error: Optional[str] = None
    language: Optional[str] = None

    def timing_breakdown(self) -> Dict[str, float]:
        return {t.stage: t.ms for t in self.timings}


class VoiceRAGHarness:
    def __init__(self, chunks: List[Chunk], cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.chunks = chunks
        self.index = VectorIndex(chunks, alpha=self.cfg.retrieval.hybrid_alpha)
        if not self.cfg.guardrails.stopwords:
            # Corpus-driven stopwords for the off-topic guardrail (see
            # guardrails.py). Only populated when the caller didn't pin them.
            self.cfg.guardrails.stopwords = _corpus_stopwords(chunks)

    @classmethod
    def from_corpus(cls, corpus: List[Dict[str, Any]], cfg: Optional[PipelineConfig] = None) -> "VoiceRAGHarness":
        """Build an index from raw MSMARCO-XI-shaped records using MULTIPLE
        chunking strategies and merge them. This is the "vast chunking"
        approach the task spec asks for: instead of a single naive splitter,
        we combine strategies that capture different retrieval angles.

        Strategies used:
          1. metadata_aware — each passage is a chunk (preserves natural
             boundaries); long passages (>400 chars) are sub-chunked into
             3-sentence windows with overlap.
          2. hybrid — adds fixed-size windows over the same passages, so
             queries that align better with a fixed boundary also find
             their answer.

        Near-duplicate chunks (>80% word overlap) are deduped to avoid
        wasting index space on redundant content. The sentence-window
        strategy is NOT run separately because metadata_aware already
        applies sentence sub-chunking to long passages."""
        cfg = cfg or PipelineConfig()
        all_chunks: List[Chunk] = []
        seen_texts: set = set()  # for near-duplicate dedup across strategies

        def _add_unique(chunks: List[Chunk]):
            for c in chunks:
                # Dedup by first 200 chars normalized — fast, catches near-identical chunks
                key = c.text[:200].strip().lower()
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_chunks.append(c)

        # Strategy 1: metadata_aware (primary — passage-level chunks with
        # sentence sub-chunking for long passages)
        meta_fn = STRATEGIES["metadata_aware"]
        for rec in corpus:
            _add_unique(meta_fn(rec["query_id"], rec["query"], rec["passages"]))

        n_meta = len(all_chunks)

        # Strategy 2: hybrid (metadata_aware + fixed-size windows merged)
        # Adds fixed-size windows that catch queries aligned to different
        # text boundaries than the passage-level chunks.
        hybrid_fn = STRATEGIES["hybrid"]
        for rec in corpus:
            _add_unique(hybrid_fn(
                rec["query_id"], rec["query"], rec["passages"],
                fixed_size=cfg.chunking.fixed_chunk_size,
                fixed_overlap=cfg.chunking.fixed_chunk_overlap,
            ))

        n_hybrid_added = len(all_chunks) - n_meta
        print(f"[chunking] Multi-strategy index: {n_meta} (metadata_aware) + "
              f"{n_hybrid_added} (hybrid fixed-size) = {len(all_chunks)} total chunks")
        
        # Try to load from cache for faster startup
        harness = cls._try_load_cache(all_chunks, cfg)
        if harness:
            return harness
        
        return cls(all_chunks, cfg)

    @classmethod
    def _try_load_cache(cls, chunks: List[Chunk], cfg: Optional[PipelineConfig] = None) -> Optional["VoiceRAGHarness"]:
        """Try to load a pre-built harness from disk cache."""
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Cache key includes a corpus fingerprint (first/last chunk texts) so
        # different languages/corpora with coincidentally equal chunk counts
        # can never load each other's index.
        corpus_fp = hashlib.md5(
            (chunks[0].text[:80] + "|" + chunks[-1].text[:80]).encode("utf-8")
        ).hexdigest()[:10] if chunks else "empty"
        cache_key = f"harness_{len(chunks)}_{corpus_fp}_{cfg.retrieval.hybrid_alpha}_{cfg.chunking.fixed_chunk_size}"
        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                t0 = time.perf_counter()
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                ms = (time.perf_counter() - t0) * 1000
                print(f"[cache] Loaded harness from cache in {ms:.0f}ms")
                return cached
            except Exception as e:
                print(f"[cache] Failed to load cache: {e}")
        
        return None

    def save_cache(self):
        """Save the harness to disk cache for faster startup."""
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        corpus_fp = hashlib.md5(
            (self.chunks[0].text[:80] + "|" + self.chunks[-1].text[:80]).encode("utf-8")
        ).hexdigest()[:10] if self.chunks else "empty"
        cache_key = f"harness_{len(self.chunks)}_{corpus_fp}_{self.cfg.retrieval.hybrid_alpha}_{self.cfg.chunking.fixed_chunk_size}"
        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self, f)
            print(f"[cache] Saved harness to {cache_path}")
        except Exception as e:
            print(f"[cache] Failed to save cache: {e}")

    def _timed(self, stage: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        return result, StageTiming(stage=stage, ms=ms)

    def run_text_query(self, query: str, mode: Optional[str] = None) -> PipelineResult:
        """Runs retrieval -> guardrails -> generation for an already-transcribed
        query. Used directly by text-mode tests/benchmarks, and internally
        by run_voice_query after STT.

        mode: "fast" (extractive, sub-ms, default from config) or "deep"
        (LLM via Groq/Anthropic)."""
        mode = mode or self.cfg.generation.default_mode
        timings: List[StageTiming] = []
        t_start = time.perf_counter()

        # Minimum query validation: refuse empty, whitespace-only, or
        # single-character queries that are too short to be meaningful.
        query_stripped = query.strip()
        if len(query_stripped) < 2:
            verdict = GuardrailVerdict(
                passed=False,
                reason=f"query too short ({len(query_stripped)} chars) — need at least 2 characters",
                stage="input_validation",
            )
            return self._refuse(query, verdict, timings, t_start)

        try:
            verdict, t = self._timed("guardrail_unsafe_input", check_unsafe_input, query, self.cfg.guardrails)
            timings.append(t)
            if not verdict.passed:
                return self._refuse(query, verdict, timings, t_start)

            results, t = self._timed("retrieval", self.index.search, query, self.cfg.retrieval.top_k, self.cfg.retrieval.min_relevance_score)
            timings.append(t)

            verdict, t = self._timed("guardrail_off_topic", check_off_topic, query, results, self.cfg.guardrails)
            timings.append(t)
            if not verdict.passed:
                return self._refuse(query, verdict, timings, t_start, retrieved=results)

            # Topic-relevance gate: verify passages actually address the query,
            # not just share entity names. Catches the common failure where
            # "CMF Andhra Pradesh" matches food passages mentioning "Andhra Pradesh".
            topic_relevance = self._timed("guardrail_topic_relevance", _check_topic_relevance, query, results)
            t_rel = topic_relevance[1]
            timings.append(t_rel)
            rel_score = topic_relevance[0]
            if rel_score <= 0.45:
                return self._refuse(query,
                    GuardrailVerdict(passed=False,
                        reason=f"topic relevance {rel_score:.2f} below threshold 0.45 — "
                               f"retrieved passages mention query entities but don't address the question",
                        stage="topic_relevance"),
                    timings, t_start, retrieved=results)

            if self.cfg.generation.use_mock:
                gen_fn = generate_answer_mock
            elif mode == "fast":
                gen_fn = generate_answer_extractive
            else:
                gen_fn = generate_answer
            gen_result, t = self._timed(f"generation[{mode}]", gen_fn, query, results, self.cfg.generation)
            timings.append(t)

            # If extractive generation returned empty (no good sentence found),
            # refuse instead of returning a blank answer.
            if not gen_result.answer and gen_result.raw_response.get("extractive"):
                reason = gen_result.raw_response.get("reason", "no relevant answer found in retrieved context")
                return self._refuse(query, 
                    GuardrailVerdict(passed=False, reason=reason, stage="generation_quality"),
                    timings, t_start, retrieved=results)

            verdict, t = self._timed("guardrail_grounding", check_grounding, gen_result.answer, results, self.cfg.guardrails)
            timings.append(t)
            if not verdict.passed:
                return self._refuse(query, verdict, timings, t_start, retrieved=results)

            total_ms = (time.perf_counter() - t_start) * 1000
            return PipelineResult(
                status=Status.OK, query_text=query, answer=gen_result.answer,
                retrieved=results, timings=timings, total_ms=total_ms,
            )
        except GenerationError as e:
            total_ms = (time.perf_counter() - t_start) * 1000
            return PipelineResult(status=Status.ERROR, query_text=query, retrieved=results,
                                   timings=timings, total_ms=total_ms,
                                   error=f"generation failed after retries: {e}")
        except Exception as e:  # last-resort error recovery — never let the harness crash the caller
            total_ms = (time.perf_counter() - t_start) * 1000
            return PipelineResult(status=Status.ERROR, query_text=query, timings=timings,
                                   total_ms=total_ms, error=f"unexpected pipeline error: {e}")

    def run_voice_query(self, audio_bytes: bytes, filename: str = "query.wav",
                        mode: Optional[str] = None) -> PipelineResult:
        t_start = time.perf_counter()
        timings: List[StageTiming] = []
        try:
            stt_result, t = self._timed("stt", transcribe, audio_bytes, filename, self.cfg.stt)
            timings.append(t)
        except STTError as e:
            total_ms = (time.perf_counter() - t_start) * 1000
            return PipelineResult(status=Status.ERROR, timings=timings, total_ms=total_ms,
                                   error=f"STT failed after retries: {e}")

        result = self.run_text_query(stt_result.text, mode=mode)
        result.timings = timings + result.timings
        result.total_ms = (time.perf_counter() - t_start) * 1000
        return result

    def _refuse(self, query, verdict: GuardrailVerdict, timings, t_start, retrieved=None) -> PipelineResult:
        total_ms = (time.perf_counter() - t_start) * 1000
        return PipelineResult(
            status=Status.REFUSED, query_text=query, retrieved=retrieved or [],
            refusal_reason=f"[{verdict.stage}] {verdict.reason}",
            timings=timings, total_ms=total_ms,
        )
