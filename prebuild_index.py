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


def prebuild(language: str, limit: int = 2000) -> None:
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
    langs = sys.argv[1:] or ["en", "hi"]
    for lang in langs:
        prebuild(lang)
