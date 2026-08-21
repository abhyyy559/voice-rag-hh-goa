"""
FastAPI entrypoint. Two ways to query:
  POST /api/query/voice  — multipart audio upload -> full voice pipeline
  POST /api/query/text   — {"query": "..."} -> retrieval+generation only, useful for
                            testing/demoing without a microphone

Run locally:
    uvicorn app.main:app --reload --port 8000
Then open http://localhost:8000

For the "live working link" submission requirement, deploy this (Render /
Railway / Fly.io / a HF Space with a Docker runtime all work) — see
TASK_HANDOFF.md for deployment notes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # Load .env before any PipelineConfig reads os.getenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline.config import PipelineConfig, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
from pipeline.harness import VoiceRAGHarness, Status
from pipeline.stt import resolve_stt_provider
from pipeline.generation import resolve_generation_provider
from data.load_dataset import load_dataset_with_fallback

app = FastAPI(title="Voice RAG — HH Goa Task 2")

_cfg = PipelineConfig()
# CORPUS_LIMIT caps the index size per language (0 = load the COMPLETE dataset).
# For fast startup, default to 500 records. Set CORPUS_LIMIT=0 for full dataset.
_corpus_limit = int(os.getenv("CORPUS_LIMIT", "500") or "500")

# Lazy-load harnesses: only build the default language at startup;
# others are built on-demand the first time they're requested.
import time as _time
import threading as _threading
import gzip as _gzip
import pickle as _pickle

t_start_init = _time.perf_counter()
_harnesses: dict[str, VoiceRAGHarness] = {}
_language_stats: dict[str, dict] = {}
_language_locks: dict[str, _threading.Lock] = {}
_language_loading: dict[str, bool] = {}

# --- Load ONLY the default language at startup for fast cold start ---
def _prebuilt_path(lang_code: str) -> str:
    """Committed, deployment-ready pickled harness (data/prebuilt/ is NOT
    vercelignored). gzip-compressed pickle loads in ~0.3s vs 5-6s to build
    the index from raw records on a cold lambda."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'prebuilt', f'{lang_code}_harness.pkl.gz')


def _load_prebuilt(lang_code: str):
    path = _prebuilt_path(lang_code)
    if not os.path.exists(path):
        return None
    t0 = _time.perf_counter()
    with open(path, 'rb') as f:
        harness = _pickle.loads(_gzip.decompress(f.read()))
    # Keep the pickled cfg (it carries the corpus-tuned stopwords), but
    # re-bind secrets from THIS environment: keys were stripped before the
    # pickle was saved so no secrets ship in git.
    harness.cfg.generation.groq_api_key = os.getenv("GROQ_API_KEY", "")
    harness.cfg.generation.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    harness.cfg.stt.sarvam_api_key = os.getenv("SARVAM_API_KEY", "")
    harness.cfg.stt.groq_api_key = os.getenv("GROQ_API_KEY", "")
    ms = (_time.perf_counter() - t0) * 1000
    print(f'[init] Loaded {lang_code} PREBUILT index: {len(harness.chunks)} chunks ({ms:.0f}ms)')
    return harness


def _load_language(lang_code: str) -> None:
    """Build a harness for a single language (thread-safe, idempotent).
    Tries prebuilt cache first for instant startup."""
    if lang_code in _harnesses:
        return
    lock = _language_locks.setdefault(lang_code, _threading.Lock())
    with lock:
        if lang_code in _harnesses:  # double-check after acquiring lock
            return
        _language_loading[lang_code] = True
        try:
            t0 = _time.perf_counter()
            # Fastest path: committed prebuilt index (deployed lambdas).
            prebuilt = _load_prebuilt(lang_code)
            if prebuilt is not None:
                _harnesses[lang_code] = prebuilt
                _language_stats[lang_code] = {'records': 0, 'chunks': len(prebuilt.chunks)}
                return
            corpus = load_dataset_with_fallback(
                prefer_real=True, limit=_corpus_limit or None, language=lang_code
            )
            if corpus:
                harness = VoiceRAGHarness.from_corpus(corpus, _cfg)
                harness.save_cache()
                _harnesses[lang_code] = harness
                _language_stats[lang_code] = {
                    "records": len(corpus),
                    "chunks": len(harness.chunks),
                }
                ms = (_time.perf_counter() - t0) * 1000
                print(f"[init] Loaded {lang_code}: {len(corpus)} records, {len(harness.chunks)} chunks ({ms:.0f}ms)")
            else:
                print(f"[init] WARNING: {lang_code} returned 0 records, skipping")
        except Exception as e:
            print(f"[init] WARNING: Failed to load {lang_code}: {e}")
        finally:
            _language_loading[lang_code] = False

# Fully lazy: don't load anything at import time. First request triggers load.
# This prevents Vercel serverless from timing out during cold start.
_harness = None  # loaded on first request
_stt_provider = resolve_stt_provider(_cfg.stt)


def _ensure_default_loaded():
    """Load the default language harness on first request (lazy init)."""
    global _harness
    if _harness is not None:
        return
    if DEFAULT_LANGUAGE not in _harnesses:
        _load_language(DEFAULT_LANGUAGE)
    _harness = _harnesses.get(DEFAULT_LANGUAGE) or next(iter(_harnesses.values()), None)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TextQuery(BaseModel):
    query: str
    mode: str = "fast"  # "fast" = extractive (sub-ms) | "deep" = LLM (Groq/Anthropic)
    language: str = DEFAULT_LANGUAGE  # ISO-639-1: "hi", "te", "bn"


def _serialize(result, mode: str = "fast") -> dict:
    return {
        "status": result.status.value,
        "query_text": result.query_text,
        "answer": result.answer,
        "refusal_reason": result.refusal_reason,
        "error": result.error,
        "mode": mode,
        "language": getattr(result, "language", DEFAULT_LANGUAGE),
        "total_ms": round(result.total_ms, 2),
        "timings": result.timing_breakdown(),
        "retrieved_sources": [
            {
                "chunk_id": r.chunk.chunk_id,
                "score": round(r.score, 4),
                "bm25_score": round(r.bm25_score, 4),
                "tfidf_score": round(r.tfidf_score, 4),
                "text": r.chunk.text[:500],
            }
            for r in result.retrieved
        ],
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    _ensure_default_loaded()
    return {
        "status": "ok",
        "stt_provider": _stt_provider,
        "generation_provider": resolve_generation_provider(_cfg.generation) or "blocked",
        "deep_ready": bool(resolve_generation_provider(_cfg.generation)),
        "default_mode": _cfg.generation.default_mode,
        "generation": "ready" if resolve_generation_provider(_cfg.generation) else "blocked (set GROQ_API_KEY or ANTHROPIC_API_KEY)",
        "languages": {
            code: {
                "name": SUPPORTED_LANGUAGES[code],
                "loaded": code in _harnesses,
                "loading": _language_loading.get(code, False),
                "records": _language_stats.get(code, {}).get("records", 0),
                "chunks": _language_stats.get(code, {}).get("chunks", 0),
            }
            for code in SUPPORTED_LANGUAGES
        },
        "default_language": DEFAULT_LANGUAGE,
    }


@app.post("/api/query/text")
def query_text(payload: TextQuery):
    if not payload.query.strip():
        raise HTTPException(400, "empty query")
    mode = payload.mode if payload.mode in ("fast", "deep") else "fast"
    lang = payload.language if payload.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    if mode == "deep" and not resolve_generation_provider(_cfg.generation):
        raise HTTPException(503, "deep mode unavailable — no GROQ_API_KEY or ANTHROPIC_API_KEY set")
    # Lazy-load on first request
    _ensure_default_loaded()
    if lang not in _harnesses:
        _load_language(lang)
    harness = _harnesses.get(lang, _harness)
    if not harness:
        raise HTTPException(503, f"language '{lang}' failed to load — try {DEFAULT_LANGUAGE}")
    result = harness.run_text_query(payload.query, mode=mode)
    result.language = lang
    return _serialize(result, mode)


@app.post("/api/query/voice")
async def query_voice(file: UploadFile = File(...), mode: str = "fast", language: str = DEFAULT_LANGUAGE):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio file")
    mode = mode if mode in ("fast", "deep") else "fast"
    lang = language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    if mode == "deep" and not resolve_generation_provider(_cfg.generation):
        raise HTTPException(503, "deep mode unavailable — no GROQ_API_KEY or ANTHROPIC_API_KEY set")
    # Lazy-load on first request
    _ensure_default_loaded()
    if lang not in _harnesses:
        _load_language(lang)
    harness = _harnesses.get(lang, _harness)
    if not harness:
        raise HTTPException(503, f"language '{lang}' failed to load — try {DEFAULT_LANGUAGE}")
    # Update STT language for voice queries
    _cfg.stt.language_code = f"{lang}-IN" if lang != "bn" else "bn-IN"
    result = harness.run_voice_query(audio_bytes, filename=file.filename or "query.wav", mode=mode)
    result.language = lang
    return _serialize(result, mode)
