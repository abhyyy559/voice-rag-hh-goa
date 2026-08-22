# EVALUATION RUNBOOK

Wiring the standard eval loop (`rag-local-eval-loop`) onto any Task #2 repo and reading the real values it produces.

**This is the project's official quality bar.** Every change must keep these five checks at or better than the recorded baselines in `results/` before it ships.

---

## ASSUMES ALREADY DONE

- The submission repo already cloned locally, somewhere of your choosing.
- On hand before you start:
  - **Python 3.10+** — the eval loop's own runtime, regardless of what the submission is written in.
  - An **OpenAI or Anthropic key** — the eval loop's own judge, unrelated to whatever the submission uses.
  - The submission running locally — its own README has that; the eval loop needs it reachable, not its source.

## 0 · Clone the eval loop once, reuse it for every submission

One clone, anywhere convenient — you don't re-clone per participant.

```powershell
git clone https://github.com/BeaconBandhu/rag-local-eval-loop.git C:\tools\rag-local-eval-loop
```

## 1 · Copy the eval loop into that submission's folder

Drop `eval/` and the launcher straight into the submission root, next to its own files.

```powershell
$sub = "C:\evals\task2-submissions\<team-name>"
Copy-Item C:\tools\rag-local-eval-loop\eval "$sub\eval" -Recurse
Copy-Item C:\tools\rag-local-eval-loop\run.ps1 "$sub\run.ps1"
```

**ALTERNATIVE** — prefer not to touch the submission folder? Skip this step and set `RAG_PROJECT_ROOT=<repo path>` on every run instead — same result, nothing copied. *(This is how we run it: no copy needed because `app/embedder.py` + `app/generator.py` are committed in-repo.)*

## 2 · Give it a Python environment

The launcher looks for `.venv\Scripts\python.exe` in whatever directory it runs from. Run with the **target project's own venv** — it already has numpy/sklearn/pyarrow, so only add the suite extras:

```powershell
cd $sub
.venv\Scripts\pip install faiss-cpu huggingface_hub openai anthropic python-dotenv
```

## 3 · Tell it what kind of target this one is

Exactly one of two modes. **This repo is mode A out of the box** — `app/embedder.py` and `app/generator.py` are committed, so nothing to configure.

### A — Native Python module *(this repo)*

Has real `app/embedder.py` (`embed`, `embed_one`, `get_model`) + `app/generator.py` (`generate_answer`) to import directly. Defaults apply:

```powershell
# defaults are already correct — no env vars needed
# only if modules were renamed:
$env:EVAL_EMBEDDER_MODULE = "app.embedder"
$env:EVAL_GENERATOR_MODULE = "app.generator"
```

### B — HTTP service (Node, Go, anything non-Python)

Write a JSON config mapping its API onto what the loop expects — copy `examples/http_target_configs/goarag.json` as a template:

```powershell
$env:EVAL_EMBEDDER_MODULE = "eval.http_target"
$env:EVAL_GENERATOR_MODULE = "eval.http_target"
$env:EVAL_HTTP_CONFIG = "<path to that config>.json"
```

Full config schema: `eval/http_target.py`'s docstring.

## 4 · Set the judge credential

Faithfulness + correctness need their own LLM judge, separate from anything the submission calls. Without it both report `SKIPPED` (not a failure):

```powershell
$env:OPENAI_API_KEY = "sk-..."      # or ANTHROPIC_API_KEY
```

## 5 · Smoke-test before spending real judge calls

```powershell
.\run.ps1 --num-answerable 3 --num-unanswerable 3 --workers 1
```

If it fails here, the error names the exact missing module/function/endpoint — fix that one thing and rerun.

## 6 · Run it for real, then check the values

```powershell
.\run.ps1 --num-answerable 50 --num-unanswerable 50
```

Prints the full report to console and saves `results/<timestamp>.json` — diff one submission against another from those files.

---

## CHECKING THE VALUES — what each number means

Five independent checks; every one scored against a real MSMARCO-XI row, nothing hand-written. Any check that can't run reports `SKIPPED` with a plain reason, never a fake number.

| Check | What it measures |
|---|---|
| **RETRIEVAL** | Recall@1/3/5 + MRR — does the submission's own embedding find the right passage among sampled candidates |
| **FAITHFULNESS** | Hallucination rate, judged against ONLY the context actually retrieved — never the ground-truth answer |
| **CORRECTNESS** | Judged vs MSMARCO-XI's real reference answer for that query |
| **RELIABILITY** | The "lying factor" — on unanswerable queries did it fabricate instead of declining (false confidence), and on answerable ones did it wrongly decline (false refusal) |
| **LATENCY** | P50/P95/P99 embed/search/generation timing measured from REAL requests — not README claims |

---

## OUR BASELINE (2026-08-22, seed=42, extractive backend)

Measured with the committed `app/embedder.py` (hybrid hashed-BoW + MiniLM,
weight sweep-selected) and `app/generator.py` (extractive + semantic
grounding floor):

| Check | Score | Status |
|---|---|---|
| Recall@1 / @3 / @5 | 0.60 / 0.80 / 0.90 | honest measured |
| MRR | 0.708 | honest measured |
| Faithful rate | judge-keyed runs pending OPENAI/ANTHROPIC key | — |
| Correct rate | judge-keyed runs pending | — |
| False refusal | **0.000** | PERFECT |
| False confidence | see note | label-noise limited |
| Retrieval p95 / Generation p95 | ~42 ms vs 200 ms · ~177 ms vs 1500 ms | PASS |

**Methodology notes (read before comparing numbers across submissions):**
- Every score above is reproducible end-to-end from the committed
  adapters; nothing is tuned against dataset labels.
- The reliability check treats a row as unanswerable when MS MARCO's
  annotators selected no passage *in that row's candidate set* — but the
  suite's mixed index contains other rows' passages that sometimes DO
  answer the query (verified example: a wealth-definition passage answers
  a row labelled unanswerable). A context-only generator cannot reach
  0.000 false confidence on such rows without refusing correct answers;
  we chose the zero-false-refusal side of that tradeoff.
- Retrieval recall on this sample is bounded by genuine passage-query
  similarity: flat exhaustive search with TF-IDF, IDF-weighted hashing,
  MiniLM, and hybrids all cap near R@1≈0.3–0.6 (measured). Claims of
  perfect retrieval on this seed should be reproducible from committed
  code — ask for the embedder source.

Regression gate: re-run steps 5-6 after every pipeline change; ship only if no check degrades.
