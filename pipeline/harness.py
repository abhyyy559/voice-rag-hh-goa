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
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

from .config import PipelineConfig
from .chunking import STRATEGIES, Chunk
from .retrieval import VectorIndex, RetrievalResult
from .guardrails import check_unsafe_input, check_off_topic, check_grounding, GuardrailVerdict
from .generation import (
    generate_answer, generate_answer_mock, generate_answer_extractive, GenerationError,
)
from .stt import transcribe, STTError, TranscriptionResult

_TOKEN_RE = re.compile(r"[a-zA-Z\u0900-\u097F]+")


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
        """Build an index from raw MSMARCO-XI-shaped records using the
        configured chunking strategy."""
        cfg = cfg or PipelineConfig()
        strategy_fn = STRATEGIES[cfg.chunking.active_strategy]
        all_chunks: List[Chunk] = []
        for rec in corpus:
            if cfg.chunking.active_strategy in ("metadata_aware", "hybrid"):
                all_chunks.extend(strategy_fn(rec["query_id"], rec["query"], rec["passages"]))
            else:
                for i, p in enumerate(rec["passages"]):
                    all_chunks.extend(strategy_fn(
                        p["text"], doc_id=f"q{rec['query_id']}_p{i}",
                        metadata={"query_id": rec["query_id"], "source": p.get("source")},
                    ))
        return cls(all_chunks, cfg)

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

        try:
            verdict, t = self._timed("guardrail_unsafe_input", check_unsafe_input, query, self.cfg.guardrails)
            timings.append(t)
            if not verdict.passed:
                return self._refuse(query, verdict, timings, t_start)

            results, t = self._timed("retrieval", self.index.search, query, self.cfg.retrieval.top_k)
            timings.append(t)

            verdict, t = self._timed("guardrail_off_topic", check_off_topic, query, results, self.cfg.guardrails)
            timings.append(t)
            if not verdict.passed:
                return self._refuse(query, verdict, timings, t_start, retrieved=results)

            if self.cfg.generation.use_mock:
                gen_fn = generate_answer_mock
            elif mode == "fast":
                gen_fn = generate_answer_extractive
            else:
                gen_fn = generate_answer
            gen_result, t = self._timed(f"generation[{mode}]", gen_fn, query, results, self.cfg.generation)
            timings.append(t)

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
