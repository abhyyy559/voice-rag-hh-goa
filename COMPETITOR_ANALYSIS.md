# Competitor Analysis — #RAGInGoa Task 2 participants

Live-tested Aug 22, 2026. Every number below was pulled from their running deployments or public repos the same day.

---

## 1. pucho.me — siddharth-09/hhgoa-task2 *(strongest competitor)*

**Live:** https://pucho.me · Repo: github.com/siddharth-09/hhgoa-task2 · AWS Docker + custom domain

| Dimension | What they built (live-verified) |
|---|---|
| Languages | Hindi + Marathi |
| Data | 20,000 queries ingested → **241,572 chunks serving** (780,881 across 4 experiment indexes, 722 MB main) |
| Embeddings | **multilingual-e5-small**, 384-dim, ONNX **int8** (`embed_query` p50 ≈ 4 ms) |
| Retrieval | Dense HNSW (hnswlib) + sparse BM25 (`bm25s`) fused with **Reciprocal Rank Fusion** |
| Answering | **Two-tier**: extractive fast answer ALWAYS computed first (p50 50 ms); optional Gemini "polish" tier OUTSIDE the 200 ms budget, verified for novel facts, falls back to the fast answer on any failure |
| Latency (their `/benchmark?n=50`, live-hit today) | **P50 60 ms · P70 64 · P90 74 · P99 79 · P100 79.5 — 50/50 within budget** |
| STT | Sarvam Saaras v3 (499–1316 ms, reported outside budget) |
| Guardrails | input-intent filter, grounding gate (abstain <0.45 support), generated-text verifier |
| Judge-facing tooling | live `/benchmark?n=` , `/compare` (runs one question through every chunking strategy side by side), requirement→evidence table, raw report files committed |
| Engineering rigor | 12 chunking variants ablated with paired-bootstrap CIs; documented a Devanagari `\w` tokenization bug (+12 % MRR when fixed); found cross-language passage-ID collisions; publishes honest caveats |

## 2. Ansh — HF Space `ansh123456789/ragingoa` *(our teammate's parallel build)*

**Live:** https://ansh123456789-ragingoa.hf.space/health (verified healthy today)

| Dimension | What they built |
|---|---|
| Languages | en, hi, mr configured; **IVF centroids precomputed for 15 Indic languages** (language routing) |
| Data | 148,545 passage-native chunks + 309 long-doc semantic chunks |
| Embeddings | **multilingual-e5-small** (intfloat) |
| Extras | semantic answer cache, query-intent filter, Sarvam STT configured, 15 s request timeout |

## 3. VasaraSujal — Hacker-House-GOA-Task-2

**Live:** https://hacker-house-goa-task-2.vercel.app (frontend) + .onrender.com backend (health verified today)

| Dimension | What they built |
|---|---|
| Languages | English-focused |
| Data | **11,478 docs** BM25 + Qdrant Cloud collection (smallest corpus of the three) |
| Embeddings | **dense via Qdrant Cloud hosted inference** (dim 384) — no local model, fits Render Free 512 MB |
| Retrieval | Qdrant dense + BM25 → **RRF → rerank** |
| STT | ElevenLabs Scribe v2 (≈545 ms, isolated from RAG timing) |
| Answers | extractive grounded synthesis (LLM tier dev-only) |
| Latency (self-reported) | RAG core P50 ≈ 158 ms — note this INCLUDES two network hops (Vercel→Render→Qdrant Cloud); still under budget |
| Judge-facing tooling | Swagger `/docs`, recommended demo flow script, keepalive so numbers are warm |

## 4. Kalp Patel's team

ElevenLabs STT, streamed-and-capped ingestion (explicitly did not load the 55.6 GB dump), grounded-only answers with sources + latency visible in the demo. LinkedIn post; no repo surfaced.

---

## Convergence: what ALL front-runners do

1. **multilingual-e5-small dense embeddings** — every single one. Semantic dense retrieval is the domain's relevance baseline; pure-lexical systems lose paraphrase/cross-lingual matches.
2. **Hybrid fusion** (dense ⊕ sparse, usually RRF) rather than one channel.
3. **Extractive-first answering** with any LLM strictly OUTSIDE the measured budget — identical philosophy to ours, arrived at independently.
4. **Self-verifiable live benchmark endpoints** so judges never have to trust a README.
5. **Warm instances** (keepalive) so measured numbers exclude cold starts.
6. **Evidence tables + raw measurement files** committed next to the claims.

---

## Head-to-head: us (voice-rag-hh-goa) vs the field

| Axis | pucho.me | Ansh | VasaraSujal | **US** |
|---|---|---|---|---|
| Pipeline latency (server-side) | 60 ms | n/a | ~158 ms incl. hops | **8-15 ms (P100 23)** ✅ fastest |
| Retrieval channels | dense+sparse RRF | dense (+centroids) | dense+sparse RRF | lexical hybrid only ⚠ gap |
| Languages shipped | 2 | 3 | 1 | **3 (en/hi/te)** ✅ |
| Chunks indexed | 241 k | 148 k | 11 k docs | **236 k across 3 langs** ✅ comparable |
| TTS answer playback | ✗ | ✗ | ✗ | **✅ browser TTS + mute** ✅ unique |
| Live self-benchmark | ✅ | ✗ | self-reports only | **✅ `/api/benchmark`** ✅ |
| Rotating real sample questions | ✗ | ✗ | static list | **✅ `/api/samples`** ✅ unique |
| Evaluator kit | partial | ✗ | swagger | **✅ runbook + adapters + stats** ✅ |
| Ablation rigor | ✅ exceptional | ✗ | ✗ | partial (benchmarks, no CI stats) ⚠ |

### Where we genuinely lead
- Raw pipeline speed (5-7× faster than pucho.me's fast path on like-for-like windows)
- Only participant with voice OUTPUT (TTS) closing the loop — it's a *voice* task
- Evaluator ergonomics: native eval-loop adapters, live benchmark, rotating real questions
- Three languages with committed, secret-free prebuilt indexes

### The one real gap (and its honest cost)
Dense semantic retrieval. Closing it properly means embedding ~236 k chunks with e5-small (hours of compute + a 140-280 MB vector store + ONNX runtime in the lambda) — pucho.me quotes 2-3 h ingest on dedicated hardware. That is a **post-deadline roadmap item** (`DECISIONS.md` §next), not a tonight patch: half-shipping a second channel hours before submission risks the working system for marginal gain on a metric judges sample manually.

---

*File generated from live endpoints hit on 2026-08-22; re-run the curls in this doc to refresh.*
