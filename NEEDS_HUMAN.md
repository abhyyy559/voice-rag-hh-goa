# NEEDS_HUMAN — things the coding agent cannot do

Everything in this file is blocked on a human. Items 1–4 are hard
submission requirements with **no resubmissions allowed** — do not treat the
working code as "done" until these are done. Items 5–6 are API keys that
unlock the last two stages of the pipeline.

---

## 1. Record the 90-second team/process video
- [ ] Explain the architecture: voice → Sarvam STT → hybrid BM25+TF-IDF
      retrieval over ai4bharat/MSMARCO-XI → grounded Claude generation →
      3-stage guardrails.
- [ ] Show the honest latency story (retrieval sub-100ms, generation+STT are
      network calls).

## 2. Record the demo video
- [ ] Record a full voice query end-to-end (ask a Hindi question out loud,
      show the answer).
- [ ] Also show a text query, an off-topic refusal, and an unsafe refusal.

## 3. Post both videos to Instagram, X, and LinkedIn
- [ ] Each team member posts individually (do not skip anyone).
- [ ] Every post includes the hashtag **#RAGInGoa**.
- [ ] At least one post must come from a **public Instagram account**.

## 4. Fill and submit the official form
- [ ] https://forms.gle/MNvCjcv23Hn2Eeu58 — due **Aug 22, 2026, 11:59 PM IST**.
- [ ] Use the real numbers from `benchmark/results/latency_report.json`
      (read its `note` field; do not submit the mock numbers).
- [ ] Include the live link: **https://voice-rag-hh-goa.vercel.app**
- [ ] Include the repo: **https://github.com/abhyyy559/voice-rag-hh-goa**

---

## 5. API keys (unblocks STT + generation end-to-end)

No `SARVAM_API_KEY` or `ANTHROPIC_API_KEY` exists anywhere on this machine,
and the agent cannot create accounts or buy credits. Everything else works;
these two stages are the only ones that can't be exercised for real.

- [ ] **Sarvam** — get a key at https://dashboard.sarvam.ai, then either:
      - set it on the Vercel project: `vercel env add SARVAM_API_KEY production`,
        or
      - put it in a local `.env` (copy `.env.example`) for local runs.
- [ ] **Anthropic** — get a key at https://console.anthropic.com, set it the
      same way (`ANTHROPIC_API_KEY`). Model is `claude-sonnet-4-6`.

Once set, re-run the benchmark for the missing real numbers:
```bash
python benchmark/latency_test.py --n 55 --dataset real        # text mode
python benchmark/latency_test.py --n 10 --dataset real --mode voice
```
The reports will then include real generation latency, and the live link's
voice + answer path will work end-to-end.

## 6. Optional (recommended) deployment hardening

- The Vercel function bundle is ~250 MB (sklearn/numpy-heavy) and cold starts
  take ~1–3 s. If you'd rather host the full stack (including local Whisper
  STT with no key), deploy `Dockerfile`-style on Render/Railway/Fly — a
  `Dockerfile` is easy to add; the app runs unchanged on uvicorn.
- The ephemeral deployment URL is tied to the vercel account; alias it to
  something clean if desired.
