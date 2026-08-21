# NEEDS_HUMAN — things the coding agent cannot do

Everything in this file is blocked on a human. **No resubmissions allowed** — do not treat the working code as "done" until these are done.

---

## 🔴 PRIORITY 1 — STT API Key (compliance requirement) — ✅ DONE

`SARVAM_API_KEY` is set in `.env` (local) AND on Vercel — live health endpoint reports `stt_provider: sarvam`. No action needed. (To change it later: `vercel env add SARVAM_API_KEY production`, then `vercel --prod`.)

---

## 🟡 PRIORITY 2 — Generation API Key — ✅ DONE

`GROQ_API_KEY` is set in `.env` AND on Vercel (`generation_provider: groq`). No action needed.

---

## 🟡 PRIORITY 3 — Re-run benchmark with fresh Groq numbers

The current benchmark numbers used a model (`llama-3.3-70b-versatile`) that Groq has since removed. Re-run after the token window resets:

```bash
# Text mode (55 queries, real corpus, real generation)
python benchmark/latency_test.py --n 55 --dataset real --corpus-limit 2000

# Voice mode (10 audio clips)
python benchmark/latency_test.py --n 10 --dataset real --mode voice
```

**Budget**: Groq free tier = ~100k tokens/day. One 55-query benchmark burns ~30-60k tokens. Do NOT re-run repeatedly — budget for ~2 runs/day max.

---

## 🟢 PRIORITY 4 — Record 2 Videos (DEADLINE: Aug 22, 11:59 PM IST)

### Video 1 — Team/Process Video (90 seconds)
- [ ] Show the architecture diagram (voice → STT → retrieval → generation)
- [ ] Explain the pipeline stages and guardrails
- [ ] Show the honest latency story (retrieval is sub-15ms, generation is a network call)
- [ ] Mention the 4 chunking strategies and 3-stage guardrails

### Video 2 — Demo Video
- [ ] Record a full voice query end-to-end (ask a Hindi question out loud, show the answer)
- [ ] Also demonstrate:
  - A text query (type and press Enter)
  - An off-topic refusal (e.g., "क्वोक्का कहाँ पाया जाता है?")
  - An unsafe refusal (e.g., "बम बनाने का तरीका बताओ")
- [ ] Show the latency breakdown in the results

---

## 🟢 PRIORITY 5 — Post Videos to Social Media (MANDATORY)

**Every team member must post individually** — not one shared team post.

For EACH team member:
- [ ] Upload both videos to **Instagram** (at least one account must be **public**)
- [ ] Upload both videos to **X (Twitter)**
- [ ] Every post on every platform must include: **#RAGInGoa**

---

## 🟢 PRIORITY 6 — Submit the Form (DEADLINE: Aug 22, 11:59 PM IST)

- [ ] Fill and submit: https://forms.gle/MNvCjcv23Hn2Eeu58
- [ ] Include:
  - GitHub repo: **https://github.com/abhyyy559/voice-rag-hh-goa**
  - Live link: **https://voice-rag-hh-goa.vercel.app**
  - Use the **real numbers** from `benchmark/results/latency_report.json` (read its `note` field)
  - Do NOT use mock numbers

---

## Summary Checklist

| # | Task | Blocked On | Status |
|---|---|---|---|
| 1 | Set SARVAM_API_KEY on Vercel | Human signup | 🔴 TODO |
| 2 | Set GROQ_API_KEY on Vercel | Human (already have locally) | 🟡 TODO |
| 3 | Re-run benchmark | API key + token window | 🟡 TODO |
| 4 | Record 2 videos | Human recording | 🟢 TODO |
| 5 | Post to Instagram + X | Human posting | 🟢 TODO |
| 6 | Submit form | Human filling form | 🟢 TODO |
