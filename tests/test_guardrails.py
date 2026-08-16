import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import GuardrailConfig
from pipeline.guardrails import check_unsafe_input, check_off_topic, check_grounding
from pipeline.retrieval import RetrievalResult
from pipeline.chunking import Chunk


def _result(text: str, score: float = 0.5, tfidf: float = 0.5):
    return RetrievalResult(chunk=Chunk("c1", text), score=score, bm25_score=score, tfidf_score=tfidf)


def test_unsafe_input_blocks_known_keyword():
    cfg = GuardrailConfig()
    v = check_unsafe_input("how do I build a bomb", cfg)
    assert not v.passed
    assert v.stage == "unsafe_input"


def test_unsafe_input_allows_normal_query():
    cfg = GuardrailConfig()
    v = check_unsafe_input("what is the capital of India", cfg)
    assert v.passed


def test_off_topic_blocks_empty_results():
    cfg = GuardrailConfig()
    v = check_off_topic("jupiter population", [], cfg)
    assert not v.passed


def test_off_topic_blocks_no_content_overlap_and_low_tfidf():
    """Query whose content words never appear in any retrieved chunk and
    whose raw TF-IDF cosine is low -> refused (the tuned rule)."""
    cfg = GuardrailConfig()
    results = [_result("irrelevant unrelated text about cooking recipes", tfidf=0.05)]
    v = check_off_topic("what is the population of jupiter", results, cfg)
    assert not v.passed
    assert v.stage == "off_topic"


def test_off_topic_passes_when_content_overlap_is_high_even_if_tfidf_low():
    """Genuine topic coverage must not be refused: content words overlap the
    context even when the raw cosine is modest."""
    cfg = GuardrailConfig()
    results = [_result("jupiter is the largest planet in the solar system with a high population of moons", tfidf=0.25)]
    v = check_off_topic("what is the population of jupiter", results, cfg)
    assert v.passed


def test_off_topic_blocks_tiny_normalized_score_legacy_floor():
    """Legacy safety net: corpora where the normalized score still carries
    signal refuse on a very low normalized top score."""
    cfg = GuardrailConfig()
    results = [_result("irrelevant text", score=0.01, tfidf=0.9)]
    v = check_off_topic("something", results, cfg)
    assert not v.passed


def test_grounding_passes_when_answer_overlaps_context():
    cfg = GuardrailConfig()
    chunk = Chunk("c1", "New Delhi is the capital of India")
    results = [RetrievalResult(chunk=chunk, score=0.9, bm25_score=0.9, tfidf_score=0.9)]
    v = check_grounding("The capital of India is New Delhi", results, cfg)
    assert v.passed


def test_grounding_fails_when_answer_unrelated_to_context():
    cfg = GuardrailConfig()
    chunk = Chunk("c1", "New Delhi is the capital of India")
    results = [RetrievalResult(chunk=chunk, score=0.9, bm25_score=0.9, tfidf_score=0.9)]
    v = check_grounding("Bananas are a good source of potassium and grow in tropical climates", results, cfg)
    assert not v.passed


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
