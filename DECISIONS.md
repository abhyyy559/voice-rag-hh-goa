# Engineering decisions (continuation log)

This file records the decisions made while completing the build, with the
measured evidence behind each one. Everything here is real and reproducible.

## 1. Dataset: real schema verified; corpus = 2000-record Hindi validation slice

`ai4bharat/MSMARCO-XI` turned out to be a per-language GeneratorBasedBuilder
whose JSONL sources have been replaced by parquet:
- Actual columns: `query`, `Answer` (capital A), `query_id`, `query_type`,
  `passages.{is_selected, English_passages, Translated_passages}`.
- The original loader's guess (`passages.passage_text`) was wrong and would
  have raised `KeyError` exactly as the handoff predicted.

Train is a single ~3.7 GB parquet (778,638 rows, one 9.7 GB row group) —
downloading it whole is impractical on slow links (~2 hours at the measured
~0.5 MB/s). The validation parquet (461,888,616 bytes, 97,941 rows) was
downloaded with 4 parallel HTTP range streams (~3 min) and is cached locally.

The shipped corpus is the first 2000 validation records (22,110 chunks),
bundled as `data/real_corpus.json` (18.6 MB) so fresh clones and serverless
deployments run against real data with no HF download at startup. `data/load_dataset.py`
tries: local parquet cache → bundled real corpus → HF streaming → sample.

## 2. Retrieval: ship the local BM25 + TF-IDF hybrid (embeddings evaluated, not shipped)

Measured retrieval quality on 120 real gold queries (recall@5, gold passage
in top-5, substring match):

| method | recall@1 | recall@3 | recall@5 |
|---|---|---|---|
| lexical hybrid (BM25+TFIDF) | 10% | 22% | 28% |
| embeddings only (MiniLM-L12 multiling.) | 2% | 22% | 26% |
| hybrid BM25 + embeddings (α=0.5) | 14% | 25% | 34% |

Embeddings give a modest gain at a large cost: index build ~145 s vs 1.4 s
(5,504 chunks), +35 ms/query, and a 470 MB model + torch that cannot ship on
the free serverless host — the deployed demo would behave differently from
the benchmarked numbers, which the handoff explicitly warns against. Decision:
**ship lexical everywhere**, document the embedding option as a swap-in point
(`VectorIndex._embed()`), and keep deployed == benchmarked == documented.

Latency: after vectorizing the top-k scoring loop, retrieval on the real
corpus measures **P50 ≈ 62–66 ms** at 22,110 chunks (BM25 ≈ 29 ms + cosine
≈ 28 ms + <2 ms overhead), P100 ≈ 240 ms. That is sub-100 ms on a real
20k+ chunk corpus — the architectural point of staying in-process holds.

## 3. STT: Sarvam request shape verified live; local Whisper fallback added

- Verified against live docs + an unauthenticated probe: endpoint
  `POST https://api.sarvam.ai/speech-to-text`, header `api-subscription-key`,
  multipart `file`/`language_code`/`model`/`mode`, response
  `{request_id, transcript, language_code}`. The probe returned a validation
  error (not 404), confirming the path, header, and field names.
- The model we previously hardcoded (`saarika:v2`) is no longer in the REST
  docs — current models are `saaras:v3` (default) / `saaras:v4`. Updated.
- Real transcription could not be tested because **no SARVAM_API_KEY exists
  in this environment** (see NEEDS_HUMAN.md). To keep the demo functional
  without a key, an `auto` provider was added: Sarvam when the key is set,
  otherwise local `faster-whisper-small` (verified: transcribes Hindi TTS
  audio to correct Devanagari in ~0.8–3 s per clip on CPU; the base model is
  too weak for Hindi — small is the floor).

## 4. Generation: endpoint shape verified; real call blocked on key

`POST https://api.anthropic.com/v1/messages` with `x-api-key` +
`anthropic-version: 2023-06-01` returns 401 with an invalid key (endpoint
exists, auth gate first). Model ID `claude-sonnet-4-6` is a current valid
Anthropic model (confirmed against docs). No `ANTHROPIC_API_KEY` exists in
this environment → **real generation latency is the one number we could not
measure** (NEEDS_HUMAN.md). Missing-key errors now fail fast (outside the
retry decorator) instead of burning 3 retries.

## 5. Guardrails: tuned against the real corpus — floors + content overlap

The min-max normalized retrieval score is ~1.0 for almost every query on a
general-domain corpus (the top hit always normalizes to 1), so the old
single floor (`off_topic_similarity_floor` on the normalized score) is
useless. Measured signals on real data:

- raw top-hit TF-IDF cosine: on-topic mean 0.46 (min 0.21) vs off-topic
  max 0.40 — partial separation.
- content-word overlap (top-3 context, stopwords stripped): separates
  better but is inflated by generic words ("व्यास", "पशु", "देश").

Fix: dual floor — refuse only when **both** raw TF-IDF < 0.35 AND
content-word overlap < 0.40. Stopwords = hand-curated Hindi/English function
words UNION corpus-frequency stopwords (tokens in ≥1% of chunks, computed at
index build). Measured against the real corpus:

- **20/20** curated no-coverage off-topic queries refused (each verified;
  the set is in `benchmark/query_sets.py`).
- **5/5** unsafe queries refused (added Hindi keywords: बम, हत्या, आत्महत्या,
  मैलवेयर, हैक, ... — the original list was English-only).
- **0 false refusals** on the corpus's own gold-retrievable queries in the
  tuning sample; 28/30 on a fresh 30-query sample — the 2 refused are noisy
  queries whose gold passage isn't lexically retrievable (safe refusal beats
  hallucination).

Honest limitation (important): MS MARCO is a general-domain corpus, so
"covered topic, unanswerable specific question" queries (e.g. "What is the
population of Jupiter?" when the corpus contains Jupiter passages) pass the
retrieval floor — that is *correct*, content exists. Those cases are handled
by the grounded-generation instruction ("if the context is insufficient, say
so") + the post-generation grounding check, which need a real LLM key to
demonstrate end-to-end.

## 6. Deployment: Vercel (the only authenticated host on this machine)

Render/Railway/Fly/HF Space all require account creation, which the agent
cannot do. Vercel CLI was already authenticated, so the app is deployed at
**https://voice-rag-hh-goa.vercel.app** with:
- `api/index.py` + `vercel.json` (single Python function, maxDuration 30 s).
- Slim `requirements.txt` (runtime deps only) — heavy dev deps
  (`datasets`, `pyarrow`, `faster-whisper`, `edge-tts`, `sentence-transformers`)
  moved to `requirements-dev.txt`.
- Verified live: `/api/health` (2000 records / 22,110 chunks), text query
  (retrieval + guardrails), off-topic refusal, unsafe refusal, demo page.
- On Vercel the whisper model isn't bundled and no API keys are set yet, so
  voice + generation return clean structured errors until the human sets
  `SARVAM_API_KEY` / `ANTHROPIC_API_KEY` on the project.

## 7. Benchmark reports

- `benchmark/results/latency_report.json` — text mode, 55 queries, real
  corpus, no mock: retrieval P50 66.12 ms, P70 82.42 ms, P100 240.0 ms;
  guardrails 20/20 off-topic refused, 5/5 unsafe refused, 28/30 on-topic
  not refused. Generation NOT RUN (no key) — stated in the report's note.
- `benchmark/results/voice_latency_report.json` — voice mode, 10 clips, real
  Whisper STT: STT P50 2,787 ms, retrieval P50 54 ms.
- Both contain a `note` field stating exactly what is real and what is not.
  No number in either file is mocked or fabricated.
