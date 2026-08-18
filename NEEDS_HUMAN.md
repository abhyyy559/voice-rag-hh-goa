# NEEDS_HUMAN — things the coding agent cannot do

Everything in this file is blocked on a human. Items 1–4 are hard
submission requirements with **no resubmissions allowed** — do not treat the
working code as "done" until these are done. Items 5–6 are API keys that
unlock the last two stages of the pipeline.

---

## 1. Record the 90-second team/process video
- [ ] Explain the architecture: voice → Sarvam STT (primary, task-spec compliant)
      → hybrid BM25+TF-IDF retrieval over ai4bharat/MSMARCO-XI → grounded
      Groq LLM generation → 3-stage guardrails.
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

The task spec requires Sarvam or ElevenLabs for STT. The code now defaults
to **Sarvam** when `SARVAM_API_KEY` is set, falling back to Groq Whisper
when only `GROQ_API_KEY` is available.

- [ ] **Set `SARVAM_API_KEY`** on the Vercel project for task-spec-compliant
      STT. Sign up at https://dashboard.sarvam.ai (free trial available).
      Then: `vercel env add SARVAM_API_KEY production` + redeploy.
- [ ] **Set `GROQ_API_KEY`** on the Vercel project for generation (+ fallback
      STT if Sarvam key is not set). Already set locally in `.env`.
      `vercel env add GROQ_API_KEY production` + redeploy.
- [ ] **Anthropic** — optional generation fallback: `ANTHROPIC_API_KEY`.

To re-run the real benchmark after the token window resets:
```bash
python benchmark/latency_test.py --n 55 --dataset real --corpus-limit 2000  # text mode
python benchmark/latency_test.py --n 10 --dataset real --mode voice
```

## 6. Optional (recommended) deployment hardening

- The Vercel function bundle is ~250 MB (sklearn/numpy-heavy) and cold starts
  take ~1–3 s. If you'd rather host the full stack (including local Whisper
  STT with no key), deploy `Dockerfile`-style on Render/Railway/Fly — a
  `Dockerfile` is easy to add; the app runs unchanged on uvicorn.
- The ephemeral deployment URL is tied to the vercel account; alias it to
  something clean if desired.
