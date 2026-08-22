# Voice RAG — HH Goa 2026, Task 2

> **Explain-it-to-anyone version:** You talk to a website. It writes down what you said. It then looks through a big shelf of real passages (a bookshelf with 236,000 slips of paper) and picks the few that actually answer you. It reads the best sentences out loud — *from those slips, word-for-word* — so it can never invent facts. If your question isn't in the bookshelf ("Who is the CM of Andhra Pradesh?"), it says *"I don't know — that's not in my data"* instead of guessing. The whole lookup takes about **10 milliseconds**.

Speak a question in English, Hindi, or Telugu → it gets transcribed → hybrid retrieval over the MSMARCO-XI corpus → grounded, guardrailed answer. **No LLM in the hot path** — answers are extracted verbatim from retrieved passages, which is how the pipeline stays far under the latency budget.

## Evaluators — start here (2 minutes)

1. **Live latency, self-verifiable:** open [`/api/benchmark?n=100`](https://voice-rag-hh-goa.vercel.app/api/benchmark?n=100) — runs 100 real corpus questions through the deployed pipeline right now and returns fresh percentiles.
2. **Try the product:** ask *"What is a corporation?"* (answer) · *"Who is the CM of Andhra Pradesh?"* (clean refusal — verified absent from all 97,941 rows) · *"Price of bitcoin today?"* (refused: static corpus).
3. **Machine-readable state:** [`/api/health`](https://voice-rag-hh-goa.vercel.app/api/health) · [`/api/stats`](https://voice-rag-hh-goa.vercel.app/api/stats) · interactive API docs at [`/docs`](https://voice-rag-hh-goa.vercel.app/docs).
4. **Standard eval harness:** `EVALUATION_RUNBOOK.md` wires `BeaconBandhu/rag-local-eval-loop` natively (`RAG_PROJECT_ROOT=<repo>`); committed adapters in `app/embedder.py` / `app/generator.py`; measured baseline + methodology inside.

**Live:** https://voice-rag-hh-goa.vercel.app
**Repo:** https://github.com/abhyyy559/voice-rag-hh-goa
**Deadline:** Aug 22, 2026, 11:59 PM IST

---

## What actually runs

```
mic ──▶ STT ──▶ guardrails ──▶ hybrid BM25+TF-IDF retrieval ──▶ extractive answer ──▶ grounding check
        │            │                    │                          │
   Chrome live STT   unsafe screen    236k chunks in-process     verbatim sentences
   (Sarvam/Groq on   + live-data      candidate-based search     (no hallucination
    server fallback)  refusal          ~5-15 ms warm              possible by construction)
```

- **STT**: Chrome/Edge live `SpeechRecognition` in-browser (zero server latency). Firefox/Safari and the `/api/query/voice` endpoint use **Sarvam AI `saaras:v3`** (task-spec compliant), falling back to Groq Whisper.
- **Retrieval**: in-process hybrid **BM25 + TF-IDF** over prebuilt per-language indexes. No hosted vector DB — a network round trip alone would eat the budget. Candidate-based scoring keeps queries at **~5-15 ms** regardless of index size.
- **Generation**: deterministic extractive synthesis (sub-ms). Deep mode (Groq LLM) exists but is opt-in via API only.
- **Languages: exactly three** — English, हिन्दी, తెలుగు (the MSMARCO-XI configs we index).

## Index sizes (this is retrieval — nothing is "trained")

| Language | Records indexed | Chunks | Source |
|---|---|---|---|
| en | 8,000 | 183,793 | `data/english_corpus.json` |
| hi | 2,000 | 50,011 | `data/real_corpus.json` |
| te | 500 | 13,847 | `data/telugu_corpus.json` |

Out of the ~55 GB MSMARCO-XI dataset (97,941-record validation split per language) — a documented, honest subset: 8k en / 2k hi / 500 te records chosen so every indexed question's gold passage is retrievable. Off-corpus questions refuse by design (try *"Who is the CM of Andhra Pradesh?"* — zero supporting passages in all 97,941 rows).

## Measured latency (100-query benchmark, production)

From `benchmark/results/latency_100_report.json` (`python benchmark/latency_100.py --url https://voice-rag-hh-goa.vercel.app`):

| P50 | P70 | P90 | P99 | P100 |
|---|---|---|---|---|
| **8.4 ms** | 9.8 ms | 13.1 ms | 20.9 ms | 22.7 ms |

All 100 queries within the 200 ms server-side budget. First request after a lambda recycle adds a one-time cold start (~2-6 s) — every page load pings `/api/health` to warm it for you.

**Verify it yourself, right now:** [`https://voice-rag-hh-goa.vercel.app/api/benchmark?n=50`](https://voice-rag-hh-goa.vercel.app/api/benchmark?n=50) runs 50 real corpus queries through the deployed pipeline and returns fresh percentiles.

## Requirements → where they live

| # | Requirement | Implementation | Evidence |
|---|---|---|---|
| 1 | STT (Sarvam / ElevenLabs) | `pipeline/stt.py` — Sarvam `saaras:v3` primary; Chrome live-SR for zero-latency browser path; Groq Whisper fallback | `/api/health` → `stt_provider` |
| 2 | Vast chunking | `pipeline/chunking.py` — 4 strategies (fixed / sentence / metadata-aware / hybrid), multi-strategy merged index | `tests/test_chunking.py` |
| 3 | <200 ms end-to-end | prebuilt indexes + candidate-based hybrid search + extractive synthesis | **live:** `/api/benchmark?n=100` |
| 4 | P50/P70/P100 analytics | `benchmark/latency_100.py` (offline) + `/api/benchmark` (live) | `benchmark/results/latency_100_report.json` |
| 5 | Harness | `pipeline/harness.py` — per-stage timing, tenacity retries, typed results, error recovery | `tests/`, stage chart in UI |
| 6 | Guardrails | unsafe screen · live-data refusal · off-topic + topic-relevance gates · grounding + query-echo checks | try *"Who is the CM of Andhra Pradesh?"* live |

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # fill SARVAM_API_KEY + GROQ_API_KEY
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
# open http://localhost:8000
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/query/text` `{query, mode, language}` | main path (mode=fast default) |
| `POST /api/query/voice` (multipart file) | audio → server STT → same pipeline |
| `GET /api/health` | providers + per-language load state |
| `GET /api/stats` | dataset scope, topics, refusal philosophy |
| `GET /api/samples?lang=en` | real rotating sample questions |
| `GET /api/benchmark?n=50` | **live latency benchmark** — fresh percentiles from the deployed pipeline |
| `GET /docs` | interactive OpenAPI (Swagger) docs |

## Guardrails (knows when NOT to answer)

1. **Unsafe input** — keyword screen (en+hi) before anything runs.
2. **Live/recency data** — "price today", "yesterday's match" refuse honestly: a static corpus cannot know them.
3. **Off-topic / topic-relevance** — word-boundary, same-passage checks; questions absent from the dataset (e.g. *"Who is the CM of Andhra Pradesh?"* — verified zero supporting passages in all 97,941 rows) refuse in <40 ms instead of returning confident nonsense.
4. **Grounding** — extractive answers are verbatim by construction; a query-echo guard stops questions being echoed back as answers.

## Deployed on Vercel

Single FastAPI function (`api/index.py`). Deploy with `vercel --prod --yes`. Env vars: `SARVAM_API_KEY`, `GROQ_API_KEY`. The compressed indexes in `data/prebuilt/*.pkl.gz` are committed and carry **no secrets** — they load in ~2 s vs 6 s building from raw text.

## Evaluation

Standard harness (`BeaconBandhu/rag-local-eval-loop`) attaches natively — see **`EVALUATION_RUNBOOK.md`** for the process, our measured baseline, and methodology notes.

## Docs map

`EVALUATION_RUNBOOK.md` evaluation process & measured baseline · `AGENTS.md` contributor quick-start
