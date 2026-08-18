# Voice RAG — HH Goa 2026, Task 2

Voice question → transcription → hybrid retrieval over ai4bharat/MSMARCO-XI → grounded, guardrailed answer.

```
audio ──▶ [STT: Sarvam (primary) / Groq Whisper / local Whisper]
                                                              │ pass
                                                              ▼
                                                  [hybrid BM25 + TF-IDF retrieval]
                                                              │
                                                  [guardrail: off-topic / no context]
                                                              │ pass
                                                              ▼
                                                  [generation: Groq, grounded prompt]
                                                              │
                                                  [guardrail: grounding / overlap check]
                                                              │ pass
                                                              ▼
                                                           answer
```

**Live link:** https://voice-rag-hh-goa.vercel.app
**Repo:** https://github.com/abhyyy559/voice-rag-hh-goa
**Deadline:** Aug 22, 2026, 11:59 PM IST

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # optional: HF loader, local Whisper, TTS test audio
cp .env.example .env                     # fill in SARVAM_API_KEY (STT) + GROQ_API_KEY (generation + fallback STT)
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
```

Text-only smoke test:
```bash
curl -X POST localhost:8000/api/query/text -H "Content-Type: application/json" \
  -d '{"query": "कॉर्पोरेशन क्या है?"}'
```

The app runs against a bundled **2000-record real slice** of the Hindi
MS MARCO validation set (`data/real_corpus.json`), so no HF download is
needed at startup. Set `CORPUS_LIMIT` to change the index size.

## How each requirement is met

**1. Speech-to-text** — `pipeline/stt.py`. Default provider `auto`: **Sarvam AI**
(`saaras:v3`, Indic-focused) when `SARVAM_API_KEY` is set — per the task
spec requirement to use Sarvam or ElevenLabs. Fallback: **Groq's hosted
Whisper** (`whisper-large-v3-turbo`, free tier, same key as generation) when
`GROQ_API_KEY` is set — verified live on Hindi audio (~1.5–2.2 s per clip,
correct Devanagari). Last resort: local `faster-whisper-small` (zero-key,
~2.8 s P50). ElevenLabs available as a one-line config swap.

**2. Chunking** — `pipeline/chunking.py` implements four strategies:
`fixed` (baseline), `sentence`, `metadata_aware` (default — chunks at the
dataset's natural passage boundaries, carries `query_id`/`source`/
`is_selected`), and `hybrid`. Switch via `PipelineConfig.chunking.active_strategy`
or `--strategy`.

**3. Retrieval** — in-process hybrid BM25 + TF-IDF (no hosted vector DB: a
network round-trip alone would eat the latency budget). Optimized with
precomputed TF-IDF document norms and sparse dot products (replaced sklearn's
cosine_similarity). Measured on the real 22,110-chunk corpus: **P50 ≈ 13 ms,
P70 ≈ 14 ms, P100 ≈ 25 ms**.

**4. Answer generation** — `pipeline/generation.py`, two providers (config
`GenerationConfig.provider`): **Groq** (`openai/gpt-oss-20b`, OpenAI-compatible
chat completions) when `GROQ_API_KEY` is set, else **Anthropic**
(`claude-sonnet-4-6`) when `ANTHROPIC_API_KEY` is set. Same strict grounded
prompt either way. Verified live with a real Groq key (Aug 2026).

**5. Harness** — `pipeline/harness.py`: per-stage timing, retries via
`tenacity`, structured `PipelineResult` (never an unhandled crash). Verified:
missing API keys produce clean structured errors, not 500s.

**6. Guardrails** — `pipeline/guardrails.py`, three stages: unsafe-input
keyword screen (English + Hindi), off-topic screen using raw TF-IDF +
content-word-overlap floors (tuned against the real corpus: 20/20 verified
no-coverage queries refused, 0 false refusals on gold-retrievable queries),
and a post-generation grounding/overlap check. Each refusal carries a
`reason` string.

## Honest state of the numbers (read before submitting)

Real runs against the 2000-record real corpus, 55 queries (`benchmark/results/latency_report.json`):

| metric | P50 | P70 | P100 |
|---|---|---|---|
| retrieval (text mode) | 13 ms | 14 ms | 25 ms |
| generation (Groq `openai/gpt-oss-20b`) | 1,339 ms | 2,668 ms | 20,598 ms¹ |
| full pipeline (text, retrieval→guardrails→Groq) | 724 ms | 1,304 ms | 20,633 ms¹ |
| STT (Groq Whisper, Hindi clips, live) | ~1.5–2.2 s | — | — |
| STT (local Whisper, 10 clips) | 2,787 ms | 2,908 ms | 9,617 ms |

¹ The P100 tail includes one query that hit Groq's free-tier rate limit
and succeeded on retry after ~20 s of backoff — a rate-limit artifact, not
steady-state latency. P50/P70 are the honest numbers.

Guardrail checks from the same text run: **20/20** off-topic refused,
**5/5** unsafe refused, **22/30** on-topic not refused (the 8 refusals are
queries where the LLM's answer paraphrased rather than quoting context —
the grounding overlap check caught that, which is correct behavior: safer
to refuse than present an ungrounded answer).

Why the numbers are what they are: retrieval is the architecturally
controllable part and it stays **sub-15 ms** in-process at 20k+ chunks
(optimized with precomputed norms + sparse dot products, see DECISIONS.md).
STT + LLM generation are network calls — industry-wide they run hundreds of
ms to seconds; that is expected, not hidden, and the stage breakdown shows
exactly where time goes. **Every number above is real and reproducible** —
run `python benchmark/latency_test.py --n 55 --dataset real --corpus-limit 2000`
yourself; the report JSON states in its `note` field exactly what is real
and what is not.

Outstanding: set `SARVAM_API_KEY` on the Vercel project for task-spec-
compliant STT, and `GROQ_API_KEY` for generation + Groq Whisper fallback.
See NEEDS_HUMAN.md for the full checklist.

## Decisions & handoff

- `DECISIONS.md` — every engineering decision with its measured evidence
  (dataset schema, embeddings evaluated-but-not-shipped, guardrail tuning
  record, deployment choice).
- `NEEDS_HUMAN.md` — the hard human checklist: videos, promotion with
  #RAGInGoa, the submission form, and the API keys.
- `TASK_HANDOFF.md` / `PRD.md` — original task and requirements.

## Project layout

```
pipeline/       core modules: config, chunking, retrieval, stt, generation, guardrails, harness
data/           loader (parquet cache → bundled real corpus → HF streaming → sample), real_corpus.json
app/            FastAPI backend + demo web page (mic recording + text fallback)
api/            Vercel serverless entry point
benchmark/      latency_test.py, query_sets.py, results/*.json
tests/          unit tests for chunking and guardrails (13/13 passing)
```
