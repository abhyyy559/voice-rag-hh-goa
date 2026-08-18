import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import STTConfig
from pipeline.stt import resolve_stt_provider, transcribe, STTError


def _cfg(**env):
    """Build an STTConfig with the given env vars set (restored after)."""
    saved = {}
    for k in ("GROQ_API_KEY", "SARVAM_API_KEY"):
        saved[k] = os.environ.get(k)
        os.environ.pop(k, None)
    try:
        for k, v in env.items():
            if v is not None:
                os.environ[k] = v
        return STTConfig()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_provider_empty_falls_back_to_whisper():
    cfg = _cfg()
    assert resolve_stt_provider(cfg) == "whisper"


def test_provider_prefers_groq_when_key_set():
    cfg = _cfg(GROQ_API_KEY="g")
    assert resolve_stt_provider(cfg) == "groq"


def test_provider_groq_beats_sarvam_in_auto():
    """With the Groq key set, auto must NOT pick Sarvam even if a Sarvam key
    also exists (Sarvam has no free tier — Groq is the free default)."""
    cfg = _cfg(GROQ_API_KEY="g", SARVAM_API_KEY="s")
    assert resolve_stt_provider(cfg) == "groq"


def test_provider_pinned_groq_requires_key():
    cfg = _cfg()
    cfg.provider = "groq"
    assert resolve_stt_provider(cfg) == "groq"  # pinned: no key check at resolve time


def test_transcribe_groq_fails_fast_without_key():
    cfg = _cfg()
    cfg.provider = "groq"
    try:
        transcribe(b"fake-audio", "q.wav", cfg)
    except STTError as e:
        assert "GROQ_API_KEY" in str(e)
    else:
        raise AssertionError("expected STTError for missing key")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError:
            print(f"FAIL {t.__name__}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
