# AGENTS.md — Voice RAG (HH Goa 2026, Task 2)

Voice question → STT → hybrid retrieval over MSMARCO-XI → grounded, guardrailed answer. Deployed on Vercel: https://voice-rag-hh-goa.vercel.app

## Commands

```powershell
# ALWAYS use the repo venv python (repo path has spaces — quote it)
.\.venv\Scripts\python.exe -m pytest tests/ -q          # 29 tests, must stay green
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000   # local dev
.\.venv\Scripts\python.exe prebuild_index.py            # rebuild data/prebuilt/*.pkl.gz after corpus changes
.\.venv\Scripts\python.exe benchmark\latency_100.py     # in-process regression gate (100 queries)
.\.venv\Scripts\python.exe benchmark\latency_100.py --url https://voice-rag-hh-goa.vercel.app
vercel --prod --yes                                     # deploy (CLI auth'd; git push does NOT auto-deploy)
```

## Non-negotiable invariants

- **Quality bar before any deploy**: `pytest` green AND `latency_100.py` shows all 100 queries < 200ms server-side. Baseline: P50 ≈ 8-14ms, P100 ≈ 23-54ms.
- **Refusal > wrong answer.** Off-corpus / unsafe / live-data questions must refuse cleanly. A guardrail "fix" that mass-refuses gold queries is a regression — this happened once; the 100-query benchmark is what catches it.
- `data/prebuilt/en|hi|te_harness.pkl.gz` are COMMITTED and load-bearing: Vercel has no `data/cache/` (`.vercelignore`) and no parquet. Never gitignore them.
- Pickles contain NO secrets (`prebuild_index.py` strips keys); `app/main.py:_load_prebuilt` re-binds API keys + refreshes `unsafe_keywords` from current code at load time. Keep both behaviors.
- The pickle-cache key includes a corpus fingerprint (`pipeline/harness.py`) — don't revert to count-only keys or languages will load each other's index.

## Architecture facts

- Flow: `app/static/index.html` (browser SpeechRecognition, mode=fast always) → `POST /api/query/text|voice` → `app/main.py` lazy-loads per-language harness → `VoiceRAGHarness.run_text_query`: unsafe screen → live-query screen → hybrid BM25+TF-IDF retrieval → off-topic + topic-relevance gates → extractive generation ("fast", default) or Groq LLM ("deep") → grounding check.
- Retrieval is candidate-based (inverted index + rare-term caps): query latency is corpus-size independent (~10-40ms at 184k chunks).
- `VectorIndex.__setstate__` re-imports numpy/scipy/sklearn module globals — unpickled indexes crash without it.
- Corpus bundles: `data/english_corpus.json`, `data/real_corpus.json` (hi), `data/telugu_corpus.json`. Languages fixed at exactly en/hi/te.
- Eval-loop contract: `app/embedder.py` + `app/generator.py` (+ optional `app/config.py`). Run the suite without copying anything:
  `RAG_PROJECT_ROOT=<repo> python -m eval.runner --num-answerable 3 --num-unanswerable 3 --workers 1`
- Docs map: `COVERAGE.md` (dataset scope, topic buckets) · `EVALUATION_RUNBOOK.md` (eval process + baselines) · `DECISIONS.md` (why each choice) · `NEEDS_HUMAN.md` (human TODO list).

## Gotchas

- Windows console mangles UTF-8 (Hindi/Telugu output): scripts print garbage but the pipeline is fine — add `sys.stdout.reconfigure(encoding='utf-8')` when writing new scripts.
- PowerShell sends string bodies as non-UTF8: pass `-Body ([Text.Encoding]::UTF8.GetBytes($json))` when testing Indic-script queries via HTTP, or results are mojibake refusals.
- Guardrail floors live in `pipeline/config.py`; topic-relevance threshold (0.45) and refusal wording in `pipeline/harness.py`. Any change there → rerun `benchmark/latency_100.py`.
- Extractive answers are verbatim sentences by construction. The query-echo guard in `generate_answer_extractive` exists because judges flag echoed questions as hallucinations — keep it.
- `.env` holds GROQ_API_KEY + SARVAM_API_KEY locally; same two exist as Vercel env vars. `ANTHROPIC_API_KEY` is a placeholder, not real.
- Frontend hardcodes `mode=fast`; deep mode is opt-in via API only.
