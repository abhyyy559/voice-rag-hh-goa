"""
Pre-build pickled VoiceRAGHarness artifacts for instant cold starts.

Why: building the index at request time takes ~4-6s per language on a
Vercel lambda (corpus load + TF-IDF/BM25 matrix build over ~13k chunks).
Unpickling a pre-built harness takes well under a second, which is what
keeps the FIRST request on a cold lambda inside a usable budget.

Artifacts land in data/prebuilt/<lang>_harness.pkl — this directory is
COMMITTED (unlike data/cache/, which .vercelignore strips) so deployed
lambdas can load them directly.

API keys are stripped before saving: the pickle must never carry secrets
(the repo is public).

Run:
    python prebuild_index.py            # en + hi
    python prebuild_index.py en         # one language
"""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()  # keys are READ here if needed, but never SAVED below

from data.load_dataset import load_dataset_with_fallback
from pipeline.config import PipelineConfig
from pipeline.harness import VoiceRAGHarness

PREBUILT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prebuilt")

_SECRET_FIELDS = ("groq_api_key", "anthropic_api_key", "sarvam_api_key",
                  "elevenlabs_api_key")


def _strip_secrets(harness: VoiceRAGHarness) -> None:
    g = harness.cfg.generation
    s = harness.cfg.stt
    for obj in (g, s):
        for f in _SECRET_FIELDS:
            if hasattr(obj, f):
                setattr(obj, f, "")


def _downcast_float32(harness: VoiceRAGHarness) -> None:
    """Halve matrix memory/disk: float64 -> float32 sparse matrices.
    Cosine/BM25 ranking precision is unaffected at float32 (guardrail
    floors sit at 0.22-0.45, float32 noise is ~1e-7 relative)."""
    import numpy as np
    idx = harness.index
    for attr in ("_count", "tfidf_matrix"):
        m = getattr(idx, attr)
        if m is not None and m.dtype == np.float64:
            setattr(idx, attr, m.astype(np.float32))
    for attr in ("_tfidf_doc_norms", "_idf", "_df", "_doc_lengths"):
        arr = getattr(idx, attr)
        if arr is not None and getattr(arr, "dtype", None) == np.float64:
            setattr(idx, attr, arr.astype(np.float32))


def prebuild(language: str, limit: int) -> None:
    t0 = time.perf_counter()
    corpus = load_dataset_with_fallback(prefer_real=True, limit=limit, language=language)
    assert corpus, f"no corpus loaded for {language}"
    cfg = PipelineConfig()
    harness = VoiceRAGHarness.from_corpus(corpus, cfg)
    build_ms = (time.perf_counter() - t0) * 1000

    # Smoke-test the index BEFORE saving: an unpicklable/broken index would
    # otherwise poison every deployed query.
    probe = harness.run_text_query("what is a corporation")
    assert probe.status.value in ("ok", "refused"), f"probe failed: {probe.error}"

    _strip_secrets(harness)
    _downcast_float32(harness)

    os.makedirs(PREBUILT_DIR, exist_ok=True)
    out = os.path.join(PREBUILT_DIR, f"{language}_harness.pkl")
    with open(out, "wb") as f:
        pickle.dump(harness, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Compress for deployment: sparse matrices shrink ~3x (65MB -> 22MB for
    # 'en') and gunzip+unpickle still loads in well under a second.
    import gzip
    gz_path = out + ".gz"
    with open(gz_path, "wb") as f:
        f.write(gzip.compress(open(out, "rb").read(), compresslevel=6))
    os.remove(out)

    size_mb = os.path.getsize(gz_path) / 1e6
    print(f"[prebuild] {language}: {len(corpus)} records -> {len(harness.chunks)} chunks | "
          f"build {build_ms:.0f}ms | saved {gz_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    # English gets the biggest slice (default judge-facing language);
    # Hindi/Telugu sized to keep the deployed bundle within limits.
    LIMITS = {"en": 8000, "hi": 2000, "te": 500}
    langs = sys.argv[1:] or ["en", "hi", "te"]
    for lang in langs:
        prebuild(lang, LIMITS.get(lang, 2000))
