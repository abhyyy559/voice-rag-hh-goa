# NEEDS_HUMAN — TODO list

Everything here is blocked on a human. **Deadline: Aug 22, 2026, 11:59 PM IST (tonight). No resubmissions allowed.**

---

## 🔴 TONIGHT — submission requirements

| # | Task | Notes |
|---|---|---|
| 1 | **Record Video 1 — team/process (90s)** | Show architecture (voice → STT → retrieval → generation), guardrails, honest latency story (`COVERAGE.md` numbers), 4 chunking strategies |
| 2 | **Record Video 2 — demo** | One voice query end-to-end · one text query · one off-corpus refusal ("Who is the CM of Andhra Pradesh?" → clean refusal) · one unsafe refusal · show latency breakdown from a response |
| 3 | **Post BOTH videos to Instagram AND X — every member individually** | At least one IG account public. Every post must include **#RAGInGoa** |
| 4 | **Submit the form**: https://forms.gle/MNvCjcv23Hn2Eeu58 | Repo: https://github.com/abhyyy559/voice-rag-hh-goa · Live: https://voice-rag-hh-goa.vercel.app · Use REAL numbers from `benchmark/results/latency_100_report.json` (P50=8.4ms P70=9.8ms P90=13.1ms P100=22.7ms server-side) |

## ✅ Done (no action)

- SARVAM_API_KEY + GROQ_API_KEY set in `.env` and on Vercel (`stt=sarvam`, `gen=groq` confirmed live)
- Production fixed: prebuilt indexes deployed, off-corpus/live-data refusals working
- 100-query benchmark: all queries <200ms server-side

## 🟡 Post-deadline technical follow-ups

- [ ] Full judge-keyed eval run (needs OPENAI_API_KEY or ANTHROPIC_API_KEY): `EVALUATION_RUNBOOK.md` §4-6 — target faithfulness ≥ 0.96 baseline
- [ ] Calibrate reliability gate further if faithfulness judge flags extractive answers (semantic cosine floor lives in `app/generator.py`)
- [ ] Corpus scale-up path: bump `LIMITS` in `prebuild_index.py`, rebuild, watch bundle size (<250MB lambda)
- [ ] Consider Vercel cron or external ping to pre-warm lambdas (cold start ~6s once per recycle)
