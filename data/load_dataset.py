"""
Loads ai4bharat/MSMARCO-XI and normalizes it into the record shape the
harness expects:
    {"query_id": ..., "query": ..., "passages": [{"text", "is_selected", "source"}, ...],
     "answer": ...}

VERIFIED against the live dataset (Aug 2026). The dataset's actual schema
differs from the standard MS MARCO shape this module originally guessed:

  - It is a GeneratorBasedBuilder with one config per language; the default
    config is "hi" (Hindi), which is what we use.
  - Passages live under `passages.Translated_passages` (not
    `passages.passage_text`), with `passages.is_selected` alongside.
  - The golden answer column is `Answer` (capital A).
  - Train is a single ~3.7GB parquet (778k rows, one 9.7GB row group);
    validation is a single ~440MB parquet (97,941 rows). Downloading the
    whole train file is impractical on slow links, so the loader prefers a
    local parquet cache (data/cache/<lang>val.parquet) and otherwise streams
    a bounded number of rows via `datasets` streaming mode.

Fallback: data/sample_data.json so the rest of the pipeline is always
runnable, even offline.
"""
import json
import os
import re
from typing import List, Dict, Any, Optional

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data.json")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
# Bundled slice of the real corpus (2000 Hindi records, generated from the
# validation parquet). Kept in the repo so deployment hosts and fresh clones
# can run against real data without downloading the 440MB parquet.
BUNDLED_REAL_PATH = os.path.join(os.path.dirname(__file__), "real_corpus.json")


def load_sample_dataset() -> List[Dict[str, Any]]:
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# File prefixes on HF are NOT simply <lang>+train/val (see the dataset
# README's language table): e.g. Hindi is `hintrain.parquet`/`hinval.parquet`.
_LANG_FILE_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def _parquet_cache_path(language: str, split: str) -> str:
    # The repo caches the downloaded parquet here. File layout on HF:
    # train/{prefix}train.parquet, validation/{prefix}val.parquet.
    prefix = _LANG_FILE_PREFIX.get(language, language)
    code = "train" if split == "train" else "val"
    return os.path.join(CACHE_DIR, f"{prefix}{code}.parquet")


def _rows_to_records(rows: List[Dict[str, Any]], english: bool = False) -> List[Dict[str, Any]]:
    """Convert raw dataset rows (either from `datasets` or pyarrow) into the
    harness record shape. english=True maps the parallel English columns
    (English_passages / Eng_Query / Eng_Answer) that ship inside every
    MSMARCO-XI parquet — the dataset was translated FROM English, so the
    originals are already there."""
    records = []
    for row in rows:
        passages_raw = row.get("passages") or {}
        # Accept both the real XI shape and the generic MS MARCO shape.
        texts = (passages_raw.get("English_passages") if english else None) \
            or passages_raw.get("Translated_passages") or passages_raw.get("passage_text")
        if texts is None:
            raise KeyError(
                "Unexpected MSMARCO-XI schema: passages has keys "
                f"{list(passages_raw.keys())}, expected 'Translated_passages' "
                "(or 'passage_text'). Inspect ds.features and update "
                "data/load_dataset.py."
            )
        selected = passages_raw.get("is_selected") or [0] * len(texts)
        qid = row.get("query_id")
        if english:
            # Eng_Query values often carry leading ". " punctuation noise.
            query = re.sub(r"^[\s.]+", "", (row.get("Eng_Query") or row.get("query") or "")).strip()
        else:
            query = row["query"]
        records.append({
            "query_id": qid,
            "query": query,
            "answer": (row.get("Eng_Answer") or row.get("Answer")) if english
                      else (row.get("Answer") or row.get("answer")),
            "passages": [
                {
                    "text": t,
                    "is_selected": int(sel) if sel is not None else 0,
                    "source": f"{'en' if english else row.get('target_lang', 'hi')}_q{qid}",
                }
                for t, sel in zip(texts, selected)
            ],
        })
    return records


def load_real_dataset(split: str = "validation", limit: Optional[int] = None,
                      language: str = "hi") -> List[Dict[str, Any]]:
    """
    Loads real data from ai4bharat/MSMARCO-XI (Hindi by default).

    `limit` is the max number of records to load; None (or 0) loads the
    COMPLETE dataset — the full 97,941-record Hindi validation parquet in
    data/cache/hinval.parquet. Set a cap (e.g. 2000) to bound index build
    time/RAM, as the deployed Vercel app must.

    Order of attempts:
      1. Local parquet cache (data/cache/<lang><split>.parquet) — fast,
         offline-capable, primary path used in this repo.
      2. `datasets` streaming load of the HF dataset — takes `limit` rows
         without downloading the full shard (slow on the 3.7GB train file,
         workable on validation).

    Raises if the schema is not the expected MS MARCO-XI shape rather than
    silently returning an empty corpus.
    """
    if language == "en":
        # English ships INSIDE the Hindi parquet: every record carries the
        # parallel English_passages / Eng_Query / Eng_Answer columns (the
        # dataset was translated from English). No separate download.
        hin = os.path.join(CACHE_DIR, "hinval.parquet")
        if os.path.exists(hin):
            return _load_from_parquet(hin, limit, english=True)
        raise RuntimeError(
            "English index needs data/cache/hinval.parquet (the English "
            "columns live inside the Hindi validation parquet)."
        )
    cache_path = _parquet_cache_path(language, split)
    # Prefer smaller subset parquets when available (faster startup)
    prefix = _LANG_FILE_PREFIX.get(language, language)
    code = "train" if split == "train" else "val"
    subset_path = os.path.join(CACHE_DIR, f"{prefix}{code}_subset.parquet")
    if os.path.exists(subset_path):
        return _load_from_parquet(subset_path, limit)
    if os.path.exists(cache_path):
        return _load_from_parquet(cache_path, limit)

    # Bundled real slice (preferred over the slow HF streaming path).
    # Only use for Hindi — this file contains Hindi-only data.
    if language == "hi" and os.path.exists(BUNDLED_REAL_PATH):
        with open(BUNDLED_REAL_PATH, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        return corpus[:limit] if limit else corpus

    try:
        from datasets import load_dataset  # local import: optional heavy dep

        try:
            # Original per-language configs are no longer resolvable on the
            # Hub (the JSONL sources were replaced by a merged 'default'
            # parquet), but try the language config first just in case.
            ds = load_dataset("ai4bharat/MSMARCO-XI", language,
                              split=split, streaming=True)
        except Exception:
            ds = load_dataset("ai4bharat/MSMARCO-XI", split=split,
                              streaming=True)
        rows = []
        for row in ds:
            if language != "default" and not str(row.get("target_lang", "")).startswith(
                    _LANG_FILE_PREFIX.get(language, language)):
                continue  # merged default config: keep only this language
            rows.append(row)
            if limit and len(rows) >= limit:
                break
        if not rows:
            raise RuntimeError(f"no rows returned for {language}/{split}")
        return _rows_to_records(rows)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ai4bharat/MSMARCO-XI ({language}/{split}): {e}. "
            f"Download validation/{_LANG_FILE_PREFIX.get(language, language)}val.parquet "
            f"into {cache_path} to use the local fast path."
        ) from e


def _load_from_parquet(path: str, limit: Optional[int], english: bool = False) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    # Path-based column selection decodes only what we need; pyarrow
    # re-nests the passages columns back into a dict.
    cols = [
        ("passages.English_passages.list.element" if english
         else "passages.Translated_passages.list.element"),
        "passages.is_selected.list.element",
        "Eng_Query" if english else "query",
        "Eng_Answer" if english else "Answer",
        "query_id", "query_type",
    ]
    table = pf.read_row_group(0, columns=cols)
    if limit:
        table = table.slice(0, limit)
    py_rows = table.to_pylist()
    return _rows_to_records(py_rows, english=english)


def load_dataset_with_fallback(prefer_real: bool = True, limit: Optional[int] = None,
                              language: str = "hi") -> List[Dict[str, Any]]:
    if prefer_real:
        try:
            return load_real_dataset(limit=limit, language=language)
        except Exception as e:
            print(f"[load_dataset] real dataset unavailable for {language} ({e}), falling back to sample_data.json")
    # Non-Hindi languages: try streaming from HF directly (no bundled fallback)
    if language != "hi":
        try:
            return load_real_dataset(limit=limit or 500, language=language)
        except Exception as e:
            print(f"[load_dataset] HF streaming failed for {language} ({e})")
            return []
    return load_sample_dataset()


if __name__ == "__main__":
    records = load_dataset_with_fallback(prefer_real=True, limit=5)
    print(f"loaded {len(records)} records")
    r = records[0]
    print("query:", r["query"])
    print("answer:", (r.get("answer") or "")[:120])
    print("n passages:", len(r["passages"]))
    print("passage[0]:", r["passages"][0]["text"][:120])
