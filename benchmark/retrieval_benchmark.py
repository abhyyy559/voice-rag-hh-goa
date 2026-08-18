"""
Measure pure retrieval latency (the full VectorIndex.search call: tokenize +
BM25 scoring + TF-IDF cosine + combined top-k) against the latency budget
defined in pipeline/config.py (PipelineConfig.latency_budget_ms, default
200ms — the PRD's sub-200ms retrieval target).

This is the repo-native adaptation of a classic "embed + FAISS search vs
budget" micro-benchmark. There is deliberately no separate embed/search
split here: this stack is in-process BM25 + TF-IDF (no embeddings, no FAISS,
no network hop — see pipeline/retrieval.py), so the whole retrieval stage is
one call. Timing it directly mirrors exactly what the harness times for its
"retrieval" stage (guardrails/generation are excluded; this measures only
the index lookup).

Queries are sampled from the corpus's own gold queries (seeded, so runs are
reproducible — same convention as benchmark/latency_test.py). Relevance does
not affect the timing, so the fallback list below is fine for corpus-less
runs.

Usage:
    python benchmark/retrieval_benchmark.py [--n 50] [--dataset real|sample]
                                            [--strategy metadata_aware]
                                            [--corpus-limit 2000]
                                            [--seed 42] [--budget 200]

Exit code 0 = PASS (p95 within budget), 1 = FAIL (see pipeline/config.py
RetrievalConfig/PipelineConfig for what to tune).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The queries are Hindi (Devanagari); make console prints UTF-8-safe on
# Windows (default cp1252 would raise UnicodeEncodeError on non-ASCII).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass  # non-stream stdout or older Python: leave as-is

from pipeline.config import PipelineConfig
from pipeline.harness import VoiceRAGHarness
from data.load_dataset import load_dataset_with_fallback

# Latency-only fallback when the corpus yields no queries (content is
# irrelevant to the measurement — any query exercises the same code path).
FALLBACK_QUERIES = [
    "कॉर्पोरेशन क्या है?",
    "मलेरिया के लक्षण क्या हैं?",
    "विटामिन डी की कमी क्या है?",
    "हृदय रोग के जोखिम कारक क्या हैं?",
    "सौर ऊर्जा कैसे काम करती है?",
]


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (k - f) * (s[c] - s[f])


def sample_queries(corpus, n, seed):
    rng = random.Random(seed)
    if not corpus:
        return list(FALLBACK_QUERIES)
    return [r["query"] for r in rng.sample(corpus, min(n, len(corpus)))]


def run_benchmark(n: int = 50, dataset: str = "real", chunking_strategy: str = "metadata_aware",
                  corpus_limit: int = 0, seed: int = 42, budget: float | None = None):
    # corpus_limit 0 = the complete dataset (full 97,941-record Hindi
    # validation parquet). Cap it to bound index build time/RAM.
    corpus = load_dataset_with_fallback(prefer_real=(dataset == "real"), limit=corpus_limit or None)
    cfg = PipelineConfig()
    cfg.chunking.active_strategy = chunking_strategy
    harness = VoiceRAGHarness.from_corpus(corpus, cfg)
    top_k = cfg.retrieval.top_k
    budget_ms = budget if budget is not None else cfg.latency_budget_ms

    queries = sample_queries(corpus, n, seed)
    n = len(queries)

    print("Warming up (index already built; first inference + imports)...")
    harness.index.search(queries[0], top_k)  # first call may pay JIT/import costs

    total_ms = []
    for i, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        harness.index.search(q, top_k)
        total_ms.append((time.perf_counter() - t0) * 1000)
        print(f"[{i:>4}/{n}] {total_ms[-1]:7.2f}ms  {q[:60]}")

    print(f"\nRan {n} queries (top_k={top_k}, strategy={chunking_strategy}, "
          f"corpus={len(corpus)} records / {len(harness.chunks)} chunks)\n")
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    print(f"{'retrieval':<12}"
          f"{mean(total_ms):>8.2f}"
          f"{percentile(total_ms, 50):>8.2f}"
          f"{percentile(total_ms, 95):>8.2f}"
          f"{percentile(total_ms, 99):>8.2f}")
    # No separate total row: retrieval IS the total here (no embed stage).

    p95_total = percentile(total_ms, 95)
    passed = p95_total <= budget_ms
    print(f"\nLatency budget: {budget_ms}ms | p95 retrieval: {p95_total:.2f}ms")
    if passed:
        print("PASS: within budget")
    else:
        print("FAIL: over budget — tune pipeline/config.py (RetrievalConfig.hybrid_alpha, "
              "ChunkingConfig chunk sizes, or raise latency_budget_ms)")

    report = {
        "n_queries": n,
        "chunking_strategy": chunking_strategy,
        "top_k": top_k,
        "corpus": {"records": len(corpus), "chunks": len(harness.chunks)},
        "latency_budget_ms": budget_ms,
        "passed": passed,
        "retrieval_ms": {
            "avg": round(mean(total_ms), 2),
            "p50": round(percentile(total_ms, 50), 2),
            "p95": round(percentile(total_ms, 95), 2),
            "p99": round(percentile(total_ms, 99), 2),
            "p100_max": round(percentile(total_ms, 100), 2),
        },
        "note": (
            f"Retrieval stage only (VectorIndex.search: BM25 + TF-IDF cosine + combined top-k), "
            f"timed exactly as the harness's 'retrieval' stage — no STT/guardrails/generation, "
            f"no network calls. Queries sampled from the corpus's own gold queries (seed={seed}). "
        ),
    }

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "results", "retrieval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report to {out_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of queries to run")
    parser.add_argument("--dataset", choices=["sample", "real"], default="real")
    parser.add_argument("--strategy", choices=["fixed", "sentence", "metadata_aware", "hybrid"],
                        default="metadata_aware")
    parser.add_argument("--corpus-limit", type=int, default=0,
                        help="max records to index (0 = complete dataset, ~98k records)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for corpus query sampling")
    parser.add_argument("--budget", type=float, default=None,
                        help="override the latency budget (default: PipelineConfig.latency_budget_ms)")
    args = parser.parse_args()
    run_benchmark(n=args.n, dataset=args.dataset, chunking_strategy=args.strategy,
                  corpus_limit=args.corpus_limit, seed=args.seed, budget=args.budget)
