# PRD — Voice-Enabled RAG Model
### HH Goa 2026 Hacker House Shortlisting, Task 2

**Owner:** [team] · **Deadline:** August 22, 2026, 11:59 PM IST · **Status:** In progress — core pipeline built & unit-tested, generation/STT/deployment pending live credentials

---

## 1. Objective

Build and ship a voice-enabled RAG system: a user speaks a question, the system transcribes it, retrieves relevant context from `ai4bharat/MSMARCO-XI`, and returns a grounded answer — end to end, in a proper harness, with guardrails, and with honest latency analytics. The goal isn't just to meet the checklist — it's to win the selection, so defensible engineering choices and transparent numbers matter more than a flashy but hand-wavy demo.

## 2. Background

This is Task Two of the HH Goa 2026 shortlisting process (Task One was a Frame/ID Card generator, handled separately). Submissions are judged on a fixed checklist (below) and require public promotional videos with a specific hashtag — those are execution/marketing requirements, not engineering ones, and are called out separately so they don't get lost.

## 3. In scope / out of scope

**In scope:** voice→text→retrieval→generation pipeline; multi-strategy chunking; hybrid local retrieval; orchestration harness with retries/structured I/O; guardrails (unsafe input, off-topic, grounding); latency benchmarking with real percentile numbers; a deployable web demo (live link); GitHub repo.

**Out of scope:** multi-turn conversation/memory, user auth, multi-language UI beyond what Sarvam's transcription already provides, fine-tuning any model, a production-grade vector DB (a hosted vector DB is explicitly avoided — see §6 rationale).

## 4. Functional requirements

| # | Requirement | Design decision | Status |
|---|---|---|---|
| 1 | Speech-to-text | Sarvam (Indic-focused, fits the Hindi/Indic MSMARCO-XI dataset better than a general-purpose provider) | Coded, **untested live** — no network route to Sarvam from the dev sandbox |
| 2 | Chunking — must be more than one naive strategy | 4 strategies implemented: fixed-size+overlap (baseline/control), sentence-window, metadata-aware (default — chunks at natural passage boundaries, carries query_id/source/is_selected metadata), hybrid (merges metadata-aware + fixed) | Built & unit-tested (5/5 passing) |
| 3 | Retrieval | Hybrid BM25 (lexical) + TF-IDF cosine (semantic-ish), in-process, no external vector DB | Built & tested — real measured latency ~1-2ms P50 |
| 4 | Answer generation | Claude (Anthropic Messages API), strict grounded-prompt instruction | Coded, **untested live** — no working API key wired in the dev sandbox |
| 5 | Harness | Central orchestrator: per-stage timing, retries via `tenacity`, structured `PipelineResult` (never an unhandled crash — failures degrade to a typed error result) | Built & validated (ran 40-query batch cleanly, 0 unhandled errors) |
| 6 | Guardrails | Pre-retrieval unsafe-keyword screen; pre-generation off-topic screen (refuses to generate if retrieval score is below a floor); post-generation grounding/overlap check | Built & unit-tested (6/6 passing); off-topic threshold needs real tuning — see §9 |
| 7 | Latency target: full pipeline < 200ms | See §6 and §9 — architecturally addressed for retrieval, **not yet achievable/measured for the full pipeline** including live STT + LLM generation | Partial — flagged as a real risk, not hidden |
| 8 | Latency analytics: P50/P70/P100 across multiple queries | `benchmark/latency_test.py` — batch runner, per-stage + full-pipeline percentiles, JSON report | Built; retrieval numbers are real, generation/STT numbers pending live run |
| 9 | Live working link | FastAPI backend + minimal mic-recording web page | Built, runs locally; **not yet deployed** |
| 10 | GitHub repo | — | Not yet pushed |
| 11 | 2 videos + promotion (IG/X/LinkedIn, `#RAGInGoa`, every team member) | — | Not started — **this is a human task, not something the coding agent can do** (see §10) |

## 5. Non-functional requirements

- **Reliability:** every pipeline failure must return a structured result, never an unhandled exception (met — verified by killing the generation API key and confirming a clean `ERROR` result).
- **Transparency over theater:** all submitted latency numbers must come from real runs, not synthetic/mocked stand-ins. Mocked benchmark runs must be clearly labeled and must never reach the submission form.
- **Reproducibility:** anyone cloning the repo, setting two env vars, and running one command should get a working demo.

## 6. Architecture & key decisions

```
audio ─▶ [STT: Sarvam] ─▶ transcript ─▶ [guardrail: unsafe input]
                                              │ pass
                                              ▼
                                  [hybrid BM25 + TF-IDF retrieval, local]
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

**Why local hybrid retrieval instead of a hosted vector DB (Pinecone/Weaviate/etc.):** a network round-trip to a hosted vector DB alone typically costs 30-150ms before any retrieval has happened, which threatens the 200ms budget before you've even started generating. In-process BM25+TF-IDF runs in low single-digit milliseconds for a corpus this size. This is the single most important latency decision in the system — it's also the reason we can hit "sub-200ms" for retrieval even though the full pipeline (with real STT + real LLM generation) almost certainly can't.

**Why Sarvam over ElevenLabs:** MSMARCO-XI is the Indic-language MS MARCO release; Sarvam is purpose-built for Indian languages, ElevenLabs is not. Swappable via one config field if this turns out wrong.

**Why TF-IDF instead of real dense embeddings for now:** the dev sandbox this was built in has no route to download embedding model weights (no HF access), so a TF-IDF+BM25 hybrid was used to keep the system fully local and testable without network access. This is a legitimate architecture, not just a fallback — but the coding agent should evaluate whether swapping in real sentence-transformer embeddings improves retrieval quality enough to justify the (likely small, still-local) latency cost, once it has network access to actually download a model.

## 7. Data

`ai4bharat/MSMARCO-XI` — not yet inspected directly (no HF access in the dev sandbox). The loader (`data/load_dataset.py`) assumes a MS MARCO-family schema (`query`, `passages.passage_text`, `passages.is_selected`) but this is a guess extrapolated from the standard MS MARCO shape, not confirmed against this specific dataset's card. **First real task for whoever has network access: load the dataset, print `.features`, and correct the loader if the schema differs.** A local 8-record sample (`data/sample_data.json`) mirrors the assumed shape and was used for all development/testing so far.

## 8. Success criteria

- All 6 technical requirements demonstrably met with real (not fabricated) evidence.
- `benchmark/results/latency_report.json` contains real P50/P70/P100 numbers from a real run (30+ queries) with real STT and real generation — even if the honest number is over 200ms, a transparent stage-by-stage breakdown is submitted rather than a suspicious single number.
- Guardrails demonstrably refuse on: unsafe input, off-topic input, and at least one contrived hallucination case.
- Live link is reachable and functional at submission time.
- Repo is clean, README accurately describes current state (not aspirational state).

## 9. Known risks / open issues (carried over honestly, not smoothed out)

1. **200ms full-pipeline target is very likely not achievable** if it includes a real STT network call and a real LLM generation call — those two alone commonly run several hundred ms to low seconds combined, industry-wide. Recommendation: report the true breakdown, call out that retrieval (the part actually in your control architecturally) is sub-5ms, and don't let the coding agent quietly fudge a "final" number to look compliant.
2. **Off-topic guardrail threshold under-tuned.** In testing, "What is the population of Jupiter?" (clearly off-topic for this corpus) was NOT refused. `GuardrailConfig.off_topic_similarity_floor` needs tuning against a real set of true-negative queries before demo day.
3. **Sarvam STT integration is unverified against live docs/endpoint.** Written from documented API shape, never called against a real endpoint.
4. **Real dataset schema is unverified** — see §7.
5. **Deployment target not chosen yet** — needs a host reachable from the public internet for the "live working link" requirement.

## 10. Explicitly a human task, not the coding agent's

The coding agent can build, test, and deploy code. It **cannot**: record a 90-second team process video, record a demo video, or post either to Instagram/X/LinkedIn from individual team members' personal accounts with `#RAGInGoa`. Flag this clearly in the handoff so it doesn't silently get treated as "done" just because the code is done — no resubmissions are allowed, so this checklist item has to be tracked by a human, separately from the build.

## 11. Timeline

- Task launch: Aug 13, 2026
- Deadline: **Aug 22, 2026, 11:59 PM**
- Recommend: code/deploy complete with 2+ days of buffer before the deadline to leave time for the video/promotion requirement, which is unrelated to build velocity and shouldn't be rushed at the last hour.
