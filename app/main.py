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

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline.config import PipelineConfig
from pipeline.harness import VoiceRAGHarness, Status
from pipeline.stt import resolve_stt_provider
from pipeline.generation import resolve_generation_provider
from data.load_dataset import load_dataset_with_fallback

app = FastAPI(title="Voice RAG — HH Goa Task 2")

_cfg = PipelineConfig()
# CORPUS_LIMIT caps the index size (0 = load the COMPLETE dataset, the full
# 97,941-record Hindi validation parquet). The deployed Vercel app MUST set
# a cap (e.g. 2000) — building the full ~1M-chunk index there OOMs/timeouts
# on serverless. See NEEDS_HUMAN.md.
_corpus_limit = int(os.getenv("CORPUS_LIMIT", "0") or "0")
_corpus = load_dataset_with_fallback(prefer_real=True, limit=_corpus_limit or None)
_harness = VoiceRAGHarness.from_corpus(_corpus, _cfg)
_stt_provider = resolve_stt_provider(_cfg.stt)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TextQuery(BaseModel):
    query: str


def _serialize(result) -> dict:
    return {
        "status": result.status.value,
        "query_text": result.query_text,
        "answer": result.answer,
        "refusal_reason": result.refusal_reason,
        "error": result.error,
        "total_ms": round(result.total_ms, 2),
        "timings": result.timing_breakdown(),
        "retrieved_sources": [
            {"chunk_id": r.chunk.chunk_id, "score": round(r.score, 4), "text": r.chunk.text[:200]}
            for r in result.retrieved
        ],
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "corpus_records": len(_corpus),
        "corpus_chunks": len(_harness.chunks),
        "stt_provider": _stt_provider,
        "generation_provider": resolve_generation_provider(_cfg.generation) or "blocked",
        "generation": "ready" if resolve_generation_provider(_cfg.generation) else "blocked (set GROQ_API_KEY or ANTHROPIC_API_KEY)",
    }


@app.post("/api/query/text")
def query_text(payload: TextQuery):
    if not payload.query.strip():
        raise HTTPException(400, "empty query")
    result = _harness.run_text_query(payload.query)
    return _serialize(result)


@app.post("/api/query/voice")
async def query_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio file")
    result = _harness.run_voice_query(audio_bytes, filename=file.filename or "query.wav")
    return _serialize(result)
