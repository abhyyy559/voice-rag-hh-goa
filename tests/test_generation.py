import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import GenerationConfig
from pipeline.generation import resolve_generation_provider, generate_answer, GenerationError


def _cfg(**env):
    """Build a GenerationConfig with the given env vars set (restored after)."""
    saved = {}
    for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        saved[k] = os.environ.get(k)
        os.environ.pop(k, None)
    try:
        for k, v in env.items():
            if v is not None:
                os.environ[k] = v
        return GenerationConfig()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_provider_empty_with_no_keys():
    cfg = _cfg()
    assert resolve_generation_provider(cfg) == ""


def test_provider_prefers_groq_when_both_keys_set():
    cfg = _cfg(GROQ_API_KEY="g", ANTHROPIC_API_KEY="a")
    assert resolve_generation_provider(cfg) == "groq"


def test_provider_falls_back_to_anthropic():
    cfg = _cfg(ANTHROPIC_API_KEY="a")
    assert resolve_generation_provider(cfg) == "anthropic"


def test_provider_pinned_groq_requires_key():
    cfg = _cfg()
    cfg.provider = "groq"
    assert resolve_generation_provider(cfg) == ""


def test_generate_answer_fails_fast_without_key():
    cfg = _cfg()
    try:
        generate_answer("कॉर्पोरेशन क्या है?", [], cfg)
    except GenerationError as e:
        assert "GROQ_API_KEY" in str(e)
    else:
        raise AssertionError("expected GenerationError for missing key")


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
