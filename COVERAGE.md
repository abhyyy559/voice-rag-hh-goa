# Corpus & Coverage Report

Answers "how much data is indexed, what can it answer, what can't it?"

## The dataset vs what we index

| Layer | Size | Records |
|---|---|---|
| ai4bharat/MSMARCO-XI — ALL languages, train+validation (HF) | ~55 GB | millions |
| Hindi validation split (`hinval.parquet`, local cache) | 462 MB | **97,941** |
| **Indexed in the deployed app (prebuilt indexes)** | ~89 MB compressed | see below |

This system is **retrieval, not training**: no model weights are trained.
The index (BM25 term-document matrix + TF-IDF vectors over text chunks) IS
the knowledge. A question is answerable **iff** passages about it exist in
the indexed slice — otherwise the guardrails refuse by design instead of
hallucinating.

| Language | Records indexed | Chunks | Share of that language's validation split |
|---|---|---|---|
| English (`en`, default) | **8,000** | **183,793** | 8.2% |
| Hindi (`hi`) | 2,000 | 50,011 | 2.0% |
| Telugu (`te`) | 500 | 13,847 | subset parquet |

Languages are deliberately limited to **three** (en/hi/te) — the dataset's
best-covered configs — so cold-start and bundle size stay within the
deployment budget.

## Topics it answers well (measured by keyword bucketing of the index)

Strong coverage (thousands of matching passages each):

- health & medicine · history & polity · business & economics
- geography & places · how-to/DIY · mathematics · science & physics
- food & cooking · education & exams · technology & computing

Thin coverage: sports, religion/culture.

Sample verified answerable queries:
- "What is a corporation?" → grounded extractive answer, ~10 ms
- "How do you make tea?" → ~5-15 ms
- "What are the symptoms of vitamin D deficiency?" → ok
- "Who wrote the national anthem?" → ok (at 8k-record index)

## What it refuses — on purpose

Verified **zero** passages in the entire 97,941-record validation set
mention e.g. Andhra Pradesh's Chief Minister / capital / CMF. Queries like:

- "Who is the CM of Andhra Pradesh?"
- "What is the capital of Andhra Pradesh?"

are **off-corpus**: retrieval finds only weak lexical matches (food or math
passages merely containing the words "Andhra Pradesh" or "cm"). The
topic-relevance guardrail measures whether retrieved sentences actually
address the question's focus (word-boundary matched, same-sentence) and
refuses at <40 ms rather than returning confident nonsense. This is the
correct RAG behavior: *knowing when not to answer*.

Also refused by design: live/recent-events questions ("who won yesterday's
match?") — a static 2018-era web QA corpus cannot know them.

## Scaling headroom (why not the full 97,941 records?)

Measured at 8,000 records / 183,793 chunks: query latency stays ~10-22 ms
(candidate-based scoring doesn't scan all chunks), but the pickled index is
64 MB gzipped and cold-load ~1.7 s. Full-corpus (~1.1 M chunks) would blow
both the serverless bundle limit and RAM. `prebuild_index.py` regenerates
the artifacts at any size — bump the LIMITS dict and redeploy.

## Reproduce everything

```bash
python prebuild_index.py            # rebuild en+hi+te prebuilt indexes
python benchmark/latency_test.py --n 55 --dataset real   # P50/P70/P100
pytest tests/                       # 25 unit/regression tests
curl localhost:8000/api/stats       # live coverage metadata
```
