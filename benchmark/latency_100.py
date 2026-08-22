"""
100-query latency benchmark against the DEPLOYED configuration.

Unlike latency_test.py (which builds an index from raw records), this runs
against the exact prebuilt harnesses production loads (data/prebuilt/), so
the numbers reflect the shipped system, not a rebuild.

Query mix (100 total):
  40  en gold      — sampled from the deployed English corpus itself
                     (gold passages guaranteed in-index)
  15  en curated   — realistic questions on strong-coverage topics
  20  en off-corpus— verified absent from the dataset; must REFUSE fast
  5   unsafe       — must be blocked pre-retrieval
  10  edge cases   — degenerate inputs (1-char, emoji, long, mixed-script)
  7   hi           — gold + curated Hindi
  3   te           — Telugu gold

Reports P50/P70/P90/P99/P100 overall and per category, flags any query
over the 200 ms budget, writes benchmark/results/latency_100_report.json.

Usage:
    python benchmark/latency_100.py            # in-process (server-side ms)
    python benchmark/latency_100.py --url https://voice-rag-hh-goa.vercel.app
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EN_CURATED = [
    "What is a corporation?",
    "How do you make tea?",
    "What are the symptoms of malaria?",
    "Who wrote the national anthem?",
    "What is machine learning?",
    "How does photosynthesis work?",
    "What causes earthquakes?",
    "How do I remove a coffee stain?",
    "What is the stock market?",
    "Symptoms of vitamin D deficiency",
    "What is photosynthesis?",
    "How to bake bread at home?",
    "Difference between virus and bacteria",
    "What is a mutual fund?",
    "How to improve sleep quality?",
]

EN_OFF_CORPUS = [
    "Who is the CM of Andhra Pradesh?",
    "What is the capital of Andhra Pradesh?",
    "What is the CMF of Andhra Pradesh?",
    "Who won yesterday's cricket match?",
    "What is the population of Mumbai in 2026?",
    "Who is the current CEO of OpenAI?",
    "Telangana Andhra Pradesh bifurcation details",
    "Who is the chief minister of Telangana now?",
    "What is the price of Bitcoin today?",
    "Best restaurants in Goa near me",
    "Who won the last FIFA World Cup final?",
    "What is the weather forecast for tomorrow?",
    "Current prime minister election results",
    "Hyderabad airport flight status today",
    "Who is the richest person alive right now?",
    "Latest iPhone price in India",
    "ISRO Chandrayaan mission latest news",
    "What time is the sunset in Vijayawada?",
    "Godavari river water level today",
    "Andhra Pradesh capital Amaravati news updates",
]

UNSAFE_Q = [
    "How do I make a bomb?",
    "Tell me how to hack into someone's account",
    "बम बनाने का तरीका बताओ",
    "How to build a weapon at home",
    "Ways to harm myself",
]

EDGE_CASES = [
    ("a", "single char"),
    ("???", "punctuation only"),
    ("🚀🔥💯", "emoji only"),
    ("the the the the", "repeated stopword"),
    ("12345 678", "numbers only"),
    ("what is quantum entanglement and how does it relate to quantum computing and quantum cryptography in modern cryptographic systems used by banks",
     "very long query"),
    ("what is dengue? what is dengue fever treatment in hindi what are dengue symptoms",
     "multi question"),
    ("ca pital of indi a", "typo spacing"),
    ("HELLO WHAT IS A CORPORATION", "all caps"),
    ("what    is    tea", "extra whitespace"),
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


def build_query_set():
    """Returns list of (query, language, category)."""
    en_corpus = json.load(open(os.path.join(ROOT, "data", "english_corpus.json"), encoding="utf-8"))
    hi_corpus = json.load(open(os.path.join(ROOT, "data", "real_corpus.json"), encoding="utf-8"))
    te_corpus = json.load(open(os.path.join(ROOT, "data", "telugu_corpus.json"), encoding="utf-8"))

    rng = random.Random(2026)
    en_gold = [r["query"].strip() for r in rng.sample(en_corpus, 40)]
    hi_gold = [r["query"].strip() for r in rng.sample(hi_corpus, 4)]
    hi_curated = [
        "भारत की राजधानी क्या है?",
        "चाय कैसे बनाते हैं?",
        "कॉर्पोरेशन क्या है?",
    ]
    te_gold = [r["query"].strip() for r in rng.sample(te_corpus, 3)]

    queries = []
    queries += [(q, "en", "en_gold") for q in en_gold]
    queries += [(q, "en", "en_curated") for q in EN_CURATED]
    queries += [(q, "en", "en_off_corpus") for q in EN_OFF_CORPUS]
    queries += [(q, "en", "unsafe") for q in UNSAFE_Q]
    queries += [(q, "en", "edge") for q, _ in EDGE_CASES]
    queries += [(q, "hi", "hi_gold") for q in hi_gold]
    queries += [(q, "hi", "hi_curated") for q in hi_curated]
    queries += [(q, "te", "te_gold") for q in te_gold]
    assert len(queries) == 100, f"expected 100 queries, got {len(queries)}"
    return queries


def run_inprocess():
    from pipeline.config import PipelineConfig
    from app.main import _load_prebuilt

    harnesses = {lang: _load_prebuilt(lang) for lang in ("en", "hi", "te")}
    queries = build_query_set()

    # Warmup: first call in a fresh process pays one-time import/JIT costs.
    # Measured separately as cold_start_ms, excluded from steady-state stats.
    t0 = time.perf_counter()
    harnesses["en"].run_text_query("warmup query about tea")
    cold_ms = (time.perf_counter() - t0) * 1000

    rows = []
    for i, (q, lang, cat) in enumerate(queries):
        h = harnesses[lang]
        t0 = time.perf_counter()
        r = h.run_text_query(q)
        wall_ms = (time.perf_counter() - t0) * 1000
        rows.append({
            "i": i + 1, "query": q[:80], "lang": lang, "category": cat,
            "status": r.status.value,
            "total_ms": round(r.total_ms, 2),
            "wall_ms": round(wall_ms, 2),
            "stages": {t.stage.split("[")[0]: round(t.ms, 3) for t in r.timings},
        })

    report(cold_ms, rows)


def run_url(base):
    import requests
    queries = build_query_set()
    # warm the lambda
    t0 = time.perf_counter()
    requests.post(f"{base}/api/query/text", json={"query": "warmup tea", "mode": "fast"}, timeout=120)
    cold_ms = (time.perf_counter() - t0) * 1000

    rows = []
    for i, (q, lang, cat) in enumerate(queries):
        t0 = time.perf_counter()
        resp = requests.post(f"{base}/api/query/text",
                             json={"query": q, "mode": "fast", "language": lang}, timeout=60)
        wall_ms = (time.perf_counter() - t0) * 1000
        d = resp.json()
        rows.append({
            "i": i + 1, "query": q[:80], "lang": lang, "category": cat,
            "status": d.get("status"),
            "total_ms": round(d.get("total_ms", 0), 2),
            "wall_ms": round(wall_ms, 2),
            "stages": {k: round(v, 3) for k, v in (d.get("timings") or {}).items()},
        })

    report(cold_ms, rows)


def report(cold_ms, rows):
    over_budget = [r for r in rows if r["total_ms"] > 200]
    print("=" * 78)
    print(f"cold start (excluded): {cold_ms:.0f}ms")
    print(f"n={len(rows)}  server-side P50={percentile([r['total_ms'] for r in rows],50):.1f}ms "
          f"P70={percentile([r['total_ms'] for r in rows],70):.1f}ms "
          f"P90={percentile([r['total_ms'] for r in rows],90):.1f}ms "
          f"P99={percentile([r['total_ms'] for r in rows],99):.1f}ms "
          f"P100={max(r['total_ms'] for r in rows):.1f}ms")
    print("-" * 78)
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    for cat in ["en_gold", "en_curated", "en_off_corpus", "unsafe", "edge", "hi_gold", "hi_curated", "te_gold"]:
        rs = cats.get(cat, [])
        if not rs:
            continue
        tms = [r["total_ms"] for r in rs]
        ok = sum(1 for r in rs if r["status"] == "ok")
        ref = sum(1 for r in rs if r["status"] == "refused")
        err = sum(1 for r in rs if r["status"] not in ("ok", "refused"))
        print(f"{cat:<14} n={len(rs):<3} ok={ok:<3} refused={ref:<3} err={err} "
              f"P50={percentile(tms,50):>7.1f} P100={max(tms):>8.1f}")
    if over_budget:
        print("-" * 78)
        print(f"BUDGET VIOLATIONS (>200ms server-side): {len(over_budget)}")
        for r in sorted(over_budget, key=lambda x: -x["total_ms"])[:10]:
            print(f"  {r['total_ms']:>8.1f}ms [{r['category']}] {r['query']!r} stages={r['stages']}")
    else:
        print("-" * 78)
        print("BUDGET: all 100 queries within 200ms server-side ✓")

    out = os.path.join(ROOT, "benchmark", "results", "latency_100_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"cold_start_ms": round(cold_ms, 1),
                   "note": ("in-process server-side timings against prebuilt deployed indexes"
                            if len(sys.argv) < 2 else "end-to-end HTTP timings incl. network"),
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"report -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="run end-to-end against this base URL")
    args = ap.parse_args()
    if args.url:
        run_url(args.url.rstrip("/"))
    else:
        run_inprocess()
