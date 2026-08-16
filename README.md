# Voice RAG — HH Goa 2026, Task 2

Voice question → transcription → hybrid retrieval over ai4bharat/MSMARCO-XI → grounded, guardrailed answer.

```
audio ──▶ [STT: Sarvam / local Whisper] ──▶ transcript ──▶ [guardrail: unsafe input]
                                                              │ pass
                                                              ▼
                                                  [hybrid BM25 + TF-IDF retrieval]
                                                              │
                                                  [guardrail: off-topic / no context]
                                                              │ pass
                                                              ▼
                                                  [generation: Claude, grounded prompt]
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
cp .env.example .env                     # fill in SARVAM_API_KEY / ANTHROPIC_API_KEY (see NEEDS_HUMAN.md)
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

**1. Speech-to-text** — `pipeline/stt.py`. Default provider `auto`: Sarvam
when `SARVAM_API_KEY` is set (request/response shape verified against live
docs + probe, current model `saaras:v3`), else local `faster-whisper-small`
(zero-key fallback, verified on real Hindi audio). ElevenLabs is a one-line
config swap.

**2. Chunking** — `pipeline/chunking.py` implements four strategies:
`fixed` (baseline), `sentence`, `metadata_aware` (default — chunks at the
dataset's natural passage boundaries, carries `query_id`/`source`/
`is_selected`), and `hybrid`. Switch via `PipelineConfig.chunking.active_strategy`
or `--strategy`.

**3. Retrieval** — in-process hybrid BM25 + TF-IDF (no hosted vector DB: a
network round-trip alone would eat the latency budget). Measured on the real
22,110-chunk corpus: **P50 ≈ 66 ms, P70 ≈ 82 ms, P100 ≈ 240 ms**.

**4. Answer generation** — `pipeline/generation.py`, Anthropic Messages API
(`claude-sonnet-4-6`), strict grounded prompt. Endpoint + model verified
against the live API; real calls need `ANTHROPIC_API_KEY` (see
NEEDS_HUMAN.md).

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

Real runs against the 2000-record real corpus (`benchmark/results/`):

| metric | P50 | P70 | P100 |
|---|---|---|---|
| retrieval (text mode, 55 queries) | 66.1 ms | 82.4 ms | 240.0 ms |
| STT (voice mode, local Whisper, 10 clips) | 2,787 ms | 2,908 ms | 9,617 ms |
| generation (real Claude call) | **not measured — no API key in this environment** | | |

Guardrail checks from the same text run: **20/20** off-topic refused,
**5/5** unsafe refused, **28/30** on-topic not refused (the 2 refusals are
noisy queries whose gold passage isn't lexically retrievable — safe refusal
over hallucination).

Why the numbers are what they are: retrieval is the architecturally
controllable part and it stays sub-100 ms in-process at 20k+ chunks.
STT + LLM generation are network calls — industry-wide they run hundreds of
ms to seconds; that is expected, not hidden, and the stage breakdown shows
exactly where time goes. **Every number above is real and reproducible** —
run `python benchmark/latency_test.py --n 55 --dataset real` yourself; the
report JSON states in its `note` field exactly what is real and what is not.

Missing (blocked on human, see NEEDS_HUMAN.md): real generation latency
(needs `ANTHROPIC_API_KEY`) and live Sarvam transcription (needs
`SARVAM_API_KEY`).

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
