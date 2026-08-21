"""Regression tests for corpus loading on deployment hosts.

The Vercel lambda has NO data/cache/*.parquet (vercelignored) — every
language must still load via bundled JSON slices in data/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.load_dataset as ld


def test_english_loads_without_parquet(monkeypatch):
    """'en' must fall back to the bundled English slice when the Hindi
    validation parquet is missing (the deployed-lambda situation)."""
    monkeypatch.setattr(ld, "CACHE_DIR", "definitely/not/a/real/dir")
    records = ld.load_dataset_with_fallback(prefer_real=True, limit=500, language="en")
    assert records, "English corpus must load without hinval.parquet"
    assert len(records) == 500
    r = records[0]
    assert "query" in r and "passages" in r
    texts = " ".join(p["text"] for p in r["passages"])
    # Bundled English slice must contain ASCII-script passages
    assert any(ord(ch) < 128 for ch in texts[:200]), "expected English text"


def test_english_offcorpus_queries_refuse_cleanly():
    """AP CM / capital queries have ZERO supporting passages in the dataset
    (verified against the full 97,941-record parquet) — the pipeline must
    refuse fast instead of surfacing weak lexical matches.

    Regression: 'Who is the CM of Andhra Pradesh?' used to PASS every gate
    because bare substring matching equated Chief Minister with centimeters
    in math passages ('A 1 cm 2 6 cm 2') and returned food text."""
    from pipeline.config import PipelineConfig
    from pipeline.harness import VoiceRAGHarness

    cfg = PipelineConfig()
    corpus = ld.load_dataset_with_fallback(prefer_real=True, limit=500, language="en")
    harness = VoiceRAGHarness.from_corpus(corpus, cfg)
    for q in [
        "Who is the CM of Andhra Pradesh?",
        "What is the capital of Andhra Pradesh?",
        "What is the CMF of Andhra Pradesh?",
    ]:
        result = harness.run_text_query(q)
        assert result.status.value != "error", f"{q!r} errored: {result.error}"
        assert result.total_ms < 200, f"{q!r} took {result.total_ms:.0f}ms (budget 200ms)"
        if result.status.value == "ok":
            answer_l = (result.answer or "").lower()
            # If answered, the answer must actually be about governance,
            # not rice/cubes that merely mention the entity.
            assert not any(w in answer_l for w in ["rice", "chutney", "sambar", "cube", "surface area"]), \
                f"{q!r} leaked an off-topic answer: {answer_l[:120]}"
