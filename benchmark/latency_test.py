"""
Runs the pipeline across a set of test queries (not a single best-case run,
per the task spec) and reports P50 / P70 / P100 latency, both for the full
pipeline and broken down per stage, so it's clear where time actually goes.

Query sets (benchmark/query_sets.py) are built against the REAL corpus:
real on-topic gold queries (sampled from the corpus at runtime), a
20-query off-topic set, and unsafe inputs.

Modes:
  --mode text   (default) run_text_query: retrieval + guardrails + generation.
  --mode voice  run_voice_query through STT (Sarvam if key present, else
                local Whisper) using the audio clips in data/audio/.

Usage:
    python benchmark/latency_test.py --n 30 --dataset real
    python benchmark/latency_test.py --n 30 --dataset real --mode voice
    python benchmark/latency_test.py --n 10 --mock-gen      # dev sanity only

*** --mock-gen is a dev-only flag: it uses a local extractive stub with NO
LLM/network call, so the numbers exclude real generation latency entirely.
It must never be used for submission numbers. ***
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

from pipeline.config import PipelineConfig
from pipeline.harness import VoiceRAGHarness, Status
from pipeline.stt import resolve_stt_provider
from data.load_dataset import load_dataset_with_fallback
from benchmark.query_sets import OFF_TOPIC, UNSAFE

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "audio")


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (k - f) * (s[c] - s[f])


def sample_on_topic_queries(corpus, n=30, seed=42):
    rng = random.Random(seed)
    return [r["query"] for r in rng.sample(corpus, min(n, len(corpus)))]


def run_benchmark(n: int = 30, dataset: str = "real", chunking_strategy: str = "metadata_aware",
                  mock_gen: bool = False, mode: str = "text", corpus_limit: int = 2000):
    corpus = load_dataset_with_fallback(prefer_real=(dataset == "real"), limit=corpus_limit)
    cfg = PipelineConfig()
    cfg.chunking.active_strategy = chunking_strategy
    cfg.generation.use_mock = mock_gen
    harness = VoiceRAGHarness.from_corpus(corpus, cfg)

    on_topic = sample_on_topic_queries(corpus, n=30)
    queries = on_topic + OFF_TOPIC + UNSAFE
    if n and n < len(queries):
        queries = queries[:n]

    all_results = []
    stage_ms: dict[str, list[float]] = {}

    audio_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith((".mp3", ".wav", ".webm"))) if mode == "voice" else []

    for i, q in enumerate(queries):
        if mode == "voice" and audio_files:
            f = audio_files[i % len(audio_files)]
            with open(os.path.join(AUDIO_DIR, f), "rb") as fh:
                result = harness.run_voice_query(fh.read(), filename=f)
        else:
            result = harness.run_text_query(q)
        all_results.append(result)
        for stage, ms in result.timing_breakdown().items():
            stage_ms.setdefault(stage, []).append(ms)
        status_tag = result.status.value.upper()
        print(f"[{status_tag:>8}] {result.total_ms:7.2f}ms  {q[:60]}")
        if result.status == Status.REFUSED:
            print(f"           -> refused: {result.refusal_reason[:120]}")
        elif result.status == Status.ERROR:
            print(f"           -> error: {result.error[:120]}")

    totals = [r.total_ms for r in all_results]
    ok_count = sum(1 for r in all_results if r.status == Status.OK)
    refused_count = sum(1 for r in all_results if r.status == Status.REFUSED)
    error_count = sum(1 for r in all_results if r.status == Status.ERROR)

    # Guardrail checks (text mode only — voice mode runs fixed audio clips,
    # so the text query sets don't apply): off-topic must be refused, unsafe
    # must be refused, on-topic must NOT be refused (they may be OK or ERROR
    # — e.g. generation is key-blocked — but refusing them is a false
    # positive of the retrieval-floor guardrail).
    if mode == "text":
        off_topic_refused = sum(1 for q, r in zip(queries, all_results)
                                if q in OFF_TOPIC and r.status == Status.REFUSED)
        unsafe_refused = sum(1 for q, r in zip(queries, all_results)
                             if q in UNSAFE and r.status == Status.REFUSED)
        on_topic_not_refused = sum(1 for q, r in zip(queries, all_results)
                                   if q in on_topic and r.status != Status.REFUSED)
        guardrail_checks = {
            "off_topic_refused": f"{off_topic_refused}/{len(OFF_TOPIC)}",
            "unsafe_refused": f"{unsafe_refused}/{len(UNSAFE)}",
            "on_topic_not_refused": f"{on_topic_not_refused}/{len(on_topic)}",
        }
    else:
        guardrail_checks = "n/a in voice mode (fixed audio clips, not the text query sets)"

    stt_provider = resolve_stt_provider(cfg.stt)

    generation_blocked = (not mock_gen and os.environ.get("ANTHROPIC_API_KEY") in (None, ""))
    note = (
        (f"*** MOCK GENERATION — DO NOT SUBMIT THESE NUMBERS *** generation used a local "
         f"extractive stub with no LLM/network call; full_pipeline_ms excludes real "
         f"generation latency entirely. "
         if mock_gen else
         ("Generation NOT RUN: ANTHROPIC_API_KEY not set in this environment, so generation "
          "rows are structured ERROR results — full_pipeline_ms covers retrieval + "
          "guardrails only. Set the key and re-run without --mock-gen for real generation "
          "latency. "
          if generation_blocked else
          "Real Anthropic API calls used for generation. "))
        + f"Corpus: {len(corpus)} records / {len(harness.chunks)} chunks (real MSMARCO-XI Hindi). "
        + f"STT provider: {stt_provider} ({mode}-mode run). "
    )

    report = {
        "n_queries": len(queries),
        "mode": mode,
        "chunking_strategy": chunking_strategy,
        "corpus": {"records": len(corpus), "chunks": len(harness.chunks)},
        "stt_provider": stt_provider,
        "mock_generation": mock_gen,
        "status_counts": {"ok": ok_count, "refused": refused_count, "error": error_count},
        "guardrail_checks": guardrail_checks,
        "full_pipeline_ms": {
            "p50": round(percentile(totals, 50), 2),
            "p70": round(percentile(totals, 70), 2),
            "p100_max": round(percentile(totals, 100), 2),
            "mean": round(mean(totals), 2) if totals else 0,
        },
        "per_stage_ms": {
            stage: {
                "p50": round(percentile(vals, 50), 2),
                "p70": round(percentile(vals, 70), 2),
                "p100_max": round(percentile(vals, 100), 2),
                "mean": round(mean(vals), 2),
            }
            for stage, vals in stage_ms.items()
        },
        "note": note,
    }

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out_name = "voice_latency_report.json" if mode == "voice" else "latency_report.json"
    out_path = os.path.join(os.path.dirname(__file__), "results", out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== LATENCY REPORT ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of test queries to run")
    parser.add_argument("--dataset", choices=["sample", "real"], default="real")
    parser.add_argument("--strategy", choices=["fixed", "sentence", "metadata_aware", "hybrid"], default="metadata_aware")
    parser.add_argument("--mock-gen", action="store_true", help="use local extractive stub instead of real LLM call (dev only, NOT for submission numbers)")
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument("--corpus-limit", type=int, default=2000)
    args = parser.parse_args()
    run_benchmark(n=args.n, dataset=args.dataset, chunking_strategy=args.strategy,
                  mock_gen=args.mock_gen, mode=args.mode, corpus_limit=args.corpus_limit)
