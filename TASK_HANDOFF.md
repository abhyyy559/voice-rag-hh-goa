# Agent Handoff Prompt — Voice RAG (HH Goa 2026, Task 2)

Copy everything below the line into your coding agent as its first instruction. It assumes the agent has this repo checked out and has (or can be given) network + API access that the environment this was built in did not have.

---

You are taking over an in-progress project: a voice-enabled RAG system for a hackathon shortlisting task, deadline **August 22, 2026, 11:59 PM IST**. Full requirements and design rationale are in `PRD.md` at the repo root — read it first, it has the full context and known risks. This prompt is your task list, not a replacement for it.

**Operating mode:** work autonomously. The human handing this off will not be reviewing each step — they've said they won't be touching this until it's done. Don't stop to ask questions; make the best reasonable engineering decision, document it (in commit messages and in `DECISIONS.md` — create it if it doesn't exist), and keep moving. The only things that should stop you are genuine hard blockers: missing API credits/keys you have no way to obtain, or something in the task spec that's actually contradictory. If you hit one of those, write it to a `NEEDS_HUMAN.md` file with a one-paragraph explanation of exactly what's blocked and what you need, and keep working on everything else that isn't blocked by it.

**Non-negotiable rule: do not fabricate or fudge benchmark numbers.** If you can't get a real number for something (e.g. no API key yet), leave it clearly marked as missing/mocked in the report — never present a synthetic number as if it were measured. A judge finding one fabricated number discredits everything else in the submission.

## Current state (as of handoff)

Already built and verified in the dev environment (no network access to HF/Sarvam/Anthropic from that sandbox, so verify these still work once you have real access — don't assume, re-run):
- Full pipeline scaffold: `pipeline/{config,chunking,retrieval,stt,generation,guardrails,harness}.py`
- 4 chunking strategies (`pipeline/chunking.py`), unit tested (`tests/test_chunking.py`, 5/5 passing)
- Hybrid BM25+TF-IDF local retrieval (`pipeline/retrieval.py`) — real measured latency ~1-2ms P50 against the 8-record sample corpus
- 3-stage guardrails (`pipeline/guardrails.py`), unit tested (`tests/test_guardrails.py`, 6/6 passing)
- Harness (`pipeline/harness.py`) — ran a 40-query batch cleanly with 0 unhandled crashes; refused correctly on an unsafe-keyword test case
- FastAPI backend + minimal mic-recording demo page (`app/main.py`, `app/static/index.html`) — boots locally, `/api/health` and `/api/query/text` both verified working
- Benchmark runner (`benchmark/latency_test.py`) with `--mock-gen` flag for dev-only sanity runs (clearly labeled, not for submission)

Explicitly NOT done / NOT verified live:
- Sarvam STT — code written from documented API shape, **never called against a real endpoint**
- Real dataset load — `data/load_dataset.py:load_real_dataset()` assumes a standard MS MARCO schema, **never actually run against `ai4bharat/MSMARCO-XI`**
- Real generation calls — `pipeline/generation.py` needs a valid `ANTHROPIC_API_KEY`; auth header is wired but untested against a live key
- Deployment — nothing is live yet
- Off-topic guardrail threshold — under-tuned, one known false negative in testing (see PRD §9)

## Task list, in order

1. **Environment setup.** `pip install -r requirements.txt`. Run `python tests/test_chunking.py` and `python tests/test_guardrails.py` — confirm 11/11 still pass before touching anything.

2. **Connect the real dataset.** Set up HF access, run `data/load_dataset.py:load_real_dataset()` against `ai4bharat/MSMARCO-XI`. It will very likely throw a `KeyError` on schema mismatch — that's expected, not a bug. Inspect `ds.features`, fix the field mapping, re-run until it loads cleanly. Re-index the harness against the real corpus (`VoiceRAGHarness.from_corpus(real_corpus, cfg)`) and sanity check a handful of queries manually before moving on.

3. **Wire and verify Sarvam STT.** Get a Sarvam API key, check `pipeline/stt.py` against current docs at https://docs.sarvam.ai (request/response schema may have changed or may simply be wrong — it was written from memory of the general shape, not the live spec). Test with at least 3 real audio clips of varying quality/accents. Fix `transcribe_sarvam()` until it reliably returns transcripts.

4. **Wire and verify generation.** Set `ANTHROPIC_API_KEY`. Run `python benchmark/latency_test.py --n 10` (no `--mock-gen`) and confirm real answers come back and look sane against the real dataset's content, not just the 8-record sample.

5. **Tune the guardrails against real data.** Expand the off-topic test set well beyond the current 10 hardcoded queries in `benchmark/latency_test.py` — include at least 15-20 true off-topic queries (about the real corpus's actual content, not just the sample's) and confirm the off-topic guardrail catches them. Adjust `GuardrailConfig.off_topic_similarity_floor` and `grounding_overlap_floor` until false negatives/positives are reasonable. Document the tuning process briefly in `DECISIONS.md`.

6. **Run the real, final latency benchmark.** `python benchmark/latency_test.py --n 30 --dataset real` (no mock) with everything real: STT, retrieval, generation. Report the honest P50/P70/P100 for the full pipeline AND the per-stage breakdown. If full-pipeline latency is well over 200ms (likely — see PRD §9, this is expected given real network calls to STT + LLM), do not hide it: present the stage breakdown showing exactly where time goes, and note in the submission that retrieval — the architecturally-controllable part — meets the target on its own. A transparent, well-reasoned "here's why and where the time goes" beats a suspicious single number.

7. **Deploy.** Pick a host reachable from the public internet (Render, Railway, Fly.io, or a Hugging Face Space with a Docker runtime all work fine for a small FastAPI app). Set the env vars (`SARVAM_API_KEY`, `ANTHROPIC_API_KEY`) on the host. Confirm the deployed `/api/health` endpoint responds and the demo page actually records + transcribes + answers over the public URL, not just localhost.

8. **Push to GitHub.** Clean history, make sure `.env` is not committed (already in `.gitignore` — double check), update `README.md`'s "Honest state of the numbers" section to reflect the actual final numbers instead of the placeholder dev-sandbox caveats.

9. **Stop and hand back to the human for these — do not attempt them yourself:**
   - Recording the 90-second team/process video
   - Recording the demo video
   - Posting both videos to Instagram, X, and LinkedIn, individually by every team member, each post including `#RAGInGoa`, with at least one public Instagram account
   - Filling and submitting https://forms.gle/MNvCjcv23Hn2Eeu58

   Write these as a clear checklist at the top of `NEEDS_HUMAN.md` even if nothing else is blocked — this is a hard requirement with no resubmissions allowed, and it's easy to treat "the code works" as "we're done" when it isn't.

## Acceptance criteria for calling this finished (from the agent's side)

- [ ] 11/11 existing unit tests still pass, plus any new tests you added for the real dataset/guardrail tuning
- [ ] `benchmark/results/latency_report.json` contains real, non-mocked P50/P70/P100 numbers from a 30+ query run against the real dataset
- [ ] Live URL is reachable and a full voice query works end to end against it
- [ ] GitHub repo is pushed, public (or shared with judges as required), README reflects real final state
- [ ] `NEEDS_HUMAN.md` clearly lists the video/promotion/submission-form steps as outstanding human tasks
