import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.chunking import fixed_size_chunks, sentence_window_chunks, metadata_aware_chunks, hybrid_chunks

SAMPLE_TEXT = ("New Delhi is the capital of India. It is the seat of all three branches "
               "of government. The city has a rich Mughal-era history. It is also a major "
               "hub for politics, culture, and commerce in North India.")


def test_fixed_size_chunks_respects_overlap():
    chunks = fixed_size_chunks(SAMPLE_TEXT, "doc1", size=50, overlap=10)
    assert len(chunks) > 1
    assert all(c.metadata["strategy"] == "fixed" for c in chunks)


def test_sentence_window_chunks_keeps_sentences_intact():
    chunks = sentence_window_chunks(SAMPLE_TEXT, "doc1", window=2, overlap=1)
    assert len(chunks) > 0
    for c in chunks:
        assert c.text.strip().endswith((".", "!", "?"))


def test_metadata_aware_chunks_attaches_query_metadata():
    passages = [{"text": SAMPLE_TEXT, "is_selected": 1, "source": "wiki"}]
    chunks = metadata_aware_chunks(query_id=1, query="capital of india", passages=passages)
    assert len(chunks) >= 1
    assert chunks[0].metadata["query_id"] == 1
    assert chunks[0].metadata["source"] == "wiki"


def test_metadata_aware_subsplits_long_passages():
    long_text = SAMPLE_TEXT * 5  # force > 400 chars
    passages = [{"text": long_text, "is_selected": 1, "source": "wiki"}]
    chunks = metadata_aware_chunks(query_id=1, query="capital of india", passages=passages)
    assert len(chunks) > 1  # should have sub-split via sentence windows


def test_hybrid_chunks_combines_strategies():
    passages = [{"text": SAMPLE_TEXT, "is_selected": 1, "source": "wiki"}]
    chunks = hybrid_chunks(query_id=1, query="capital of india", passages=passages)
    strategies_present = {c.metadata.get("strategy") for c in chunks}
    assert "metadata_aware" in strategies_present or "sentence" in strategies_present
    assert "fixed" in strategies_present


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
