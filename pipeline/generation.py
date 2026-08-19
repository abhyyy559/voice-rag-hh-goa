"""
Answer generation stage. Three providers (config.py -> GenerationConfig.provider):
  - "extractive": Deterministic, sub-ms answer synthesis from the retrieved
                  passages (no network call) — the default fast path that keeps
                  the RAG pipeline under the 200ms latency target. Grounded by
                  construction: every returned sentence comes verbatim from a
                  retrieved passage.
  - "groq"      : Groq's OpenAI-compatible chat completions API (deep mode;
                  fast, cheap, verified live Aug 2026).
  - "anthropic" : Anthropic Messages API (claude-sonnet-4-6), used when no
                  GROQ_API_KEY but ANTHROPIC_API_KEY is set.
The LLM providers use the same strict grounding instruction so the model
prefers "not enough information" over guessing — the
guardrails.check_grounding step then double-checks that behaviorally instead
of just trusting the prompt.
"""
import re
import requests
from dataclasses import dataclass
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import GenerationConfig
from .retrieval import RetrievalResult

SYSTEM_PROMPT = (
    "You are a retrieval-grounded QA assistant. Answer ONLY using the provided "
    "context passages. If the context does not contain enough information to "
    "answer confidently, say so explicitly instead of guessing. Be concise — "
    "2-4 sentences. Do not invent facts not present in the context."
)


@dataclass
class GenerationResult:
    answer: str
    raw_response: dict


def resolve_generation_provider(cfg: GenerationConfig) -> str:
    """Resolves the DEEP (LLM) provider: "auto" -> groq -> anthropic by which
    key is set; explicit "groq"/"anthropic" pin the provider (empty string =
    no key available — deep mode unavailable, fast/extractive mode unaffected).
    The fast path (extractive) is chosen by GenerationConfig.default_mode /
    the per-request mode, not by this function."""
    if cfg.use_mock:
        return "mock"
    if cfg.provider == "groq":
        return "groq" if cfg.groq_api_key else ""
    if cfg.provider == "anthropic":
        return "anthropic" if cfg.anthropic_api_key else ""
    if cfg.groq_api_key:
        return "groq"
    if cfg.anthropic_api_key:
        return "anthropic"
    return ""


def _build_prompt(query: str, results: List[RetrievalResult]) -> str:
    context_block = "\n\n".join(
        f"[{i+1}] {r.chunk.text}" for i, r in enumerate(results)
    )
    return (
        f"Context passages:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above. If it's insufficient, say so."
    )


class GenerationError(Exception):
    pass


class RateLimitedError(GenerationError):
    """The API returned 429 (rate limit). Raised so the retry layer can back
    off for longer than a transient network error — Groq's free tier is
    ~1000 req/hour and ~12k tokens/minute, so bursts WILL hit this."""

    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


def _generation_wait(retry_state) -> float:
    """Custom tenacity wait: honor the API's Retry-After hint on 429s
    (with a floor so we don't hammer), else exponential backoff."""
    exc = retry_state.outcome.exception()
    if isinstance(exc, RateLimitedError) and exc.retry_after > 0:
        return max(exc.retry_after, 4.0)
    return min(2 ** (retry_state.attempt_number - 1), 10.0)


def generate_answer_mock(query: str, results: List[RetrievalResult], cfg: GenerationConfig) -> GenerationResult:
    """
    Extractive stand-in for local dev/testing ONLY — returns the top retrieved
    chunk's text instead of calling an LLM. No network call, so it lets the
    rest of the harness (retrieval, guardrails, timing) be exercised without
    API access.

    *** DO NOT use figures produced with this stub as your submitted latency
    numbers. *** It has none of the real generation latency (typically the
    single largest chunk of end-to-end time). Swap back to generate_answer()
    and re-run benchmark/latency_test.py with a real ANTHROPIC_API_KEY before
    recording numbers for submission.
    """
    if not results:
        return GenerationResult(answer="", raw_response={"mock": True})
    answer = results[0].chunk.text
    return GenerationResult(answer=answer, raw_response={"mock": True, "source_chunk": results[0].chunk.chunk_id})


# ---------------------------------------------------------------------------
# Fast path: deterministic extractive answer synthesis (sub-ms, no network).
# This is what keeps the RAG pipeline under the 200ms latency target: every
# returned sentence is lifted verbatim from a retrieved passage, so the answer
# is grounded by construction and always passes the grounding guardrail.
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z\u0900-\u097F]+")


def _content_words(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]


def generate_answer_extractive(query: str, results: List[RetrievalResult], cfg: GenerationConfig) -> GenerationResult:
    """Deterministic extractive QA: scores every sentence in the top retrieved
    passages by content-word overlap with the query and returns the best 1-3
    sentences (highest scoring first, capped by max_sentences). Falls back to
    the lead sentence of the top passage when no sentence has any overlap —
    the passage was already deemed relevant by retrieval + the off-topic
    guardrail, so its lead sentence is the safest grounded summary."""
    max_sentences = getattr(cfg, "extractive_max_sentences", 3)
    min_sentence_words = 3   # skip fragments like "See also." / "वेबसाइट।"
    if not results:
        return GenerationResult(answer="", raw_response={"extractive": True})

    q_words = _content_words(query)
    scored: List[tuple] = []  # (score, passage_rank, sentence_idx, sentence)
    for rank, r in enumerate(results[:3]):  # top-3 passages only — keeps it sub-ms
        for si, sent in enumerate(_split_sentences(r.chunk.text)):
            words = _WORD_RE.findall(sent)
            if len(words) < min_sentence_words:
                continue
            overlap = len(q_words & set(w.lower() for w in words))
            # small passage-rank tiebreak: earlier passages win ties
            scored.append((overlap + 1e-6 * (len(results) - rank), rank, si, sent))

    if not scored:
        lead = _split_sentences(results[0].chunk.text)
        answer = lead[0] if lead else results[0].chunk.text[:300]
        return GenerationResult(answer=answer, raw_response={"extractive": True, "fallback": "lead"})

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    picked = scored[:max_sentences]
    # present in passage order (reading flow) rather than score order
    picked.sort(key=lambda t: (t[1], t[2]))
    answer = " ".join(s for *_, s in picked)
    return GenerationResult(
        answer=answer,
        raw_response={"extractive": True, "sentences": [s for *_, s in picked]},
    )


@retry(stop=stop_after_attempt(4), wait=_generation_wait,
       retry=retry_if_exception_type(GenerationError), reraise=True)
def _generate_answer_anthropic_once(query: str, results: List[RetrievalResult], cfg: GenerationConfig,
                                    api_key: str, prompt: str) -> GenerationResult:
    try:
        resp = requests.post(
            cfg.api_url,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": cfg.model,
                "max_tokens": cfg.max_tokens,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=cfg.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise GenerationError(str(e)) from e

    text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    answer = "\n".join(text_blocks).strip()
    if not answer:
        raise GenerationError("empty response from generation API")
    return GenerationResult(answer=answer, raw_response=data)


@retry(stop=stop_after_attempt(4), wait=_generation_wait,
       retry=retry_if_exception_type(GenerationError), reraise=True)
def _generate_answer_groq_once(query: str, results: List[RetrievalResult], cfg: GenerationConfig,
                               api_key: str, prompt: str) -> GenerationResult:
    """Groq's OpenAI-compatible chat completions endpoint. Request/response
    shape is the standard chat.completion object: choices[0].message.content."""
    try:
        resp = requests.post(
            cfg.groq_endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.groq_model,
                "max_tokens": cfg.max_tokens,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=cfg.timeout_s,
        )
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "0") or 0)
            raise RateLimitedError(f"Groq rate limit (429), retry after {retry_after}s", retry_after=retry_after)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise GenerationError(str(e)) from e

    try:
        answer = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise GenerationError(f"unexpected Groq response shape: {str(data)[:200]}") from e
    if not answer:
        raise GenerationError("empty response from generation API")
    return GenerationResult(answer=answer, raw_response=data)


def generate_answer(query: str, results: List[RetrievalResult], cfg: GenerationConfig) -> GenerationResult:
    """Dispatches to the configured provider. Missing-key errors are checked
    OUTSIDE the retried inner calls so a missing key fails fast (single
    attempt) instead of burning 3 retries."""
    provider = resolve_generation_provider(cfg)
    if provider == "mock":
        return generate_answer_mock(query, results, cfg)
    if not provider:
        raise GenerationError(
            "no generation API key set — set GROQ_API_KEY (or ANTHROPIC_API_KEY) in the environment"
        )
    prompt = _build_prompt(query, results)
    if provider == "groq":
        return _generate_answer_groq_once(query, results, cfg, cfg.groq_api_key, prompt)
    return _generate_answer_anthropic_once(query, results, cfg, cfg.anthropic_api_key, prompt)
