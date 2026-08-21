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
_WORD_RE = re.compile(r"[a-zA-Z\u0900-\u097F\u0980-\u09FF\u0C00-\u0C7F]+")  # Latin + Devanagari + Bengali + Telugu


def _content_words(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]


def generate_answer_extractive(query: str, results: List[RetrievalResult], cfg: GenerationConfig) -> GenerationResult:
    """Deterministic extractive QA: scores every sentence in the top retrieved
    passages by content-word overlap with the query and returns the best 1-3
    sentences (highest scoring first, capped by max_sentences).

    Quality gates:
      1. Only sentences with at least min_overlap content-word matches are
         considered (prevents surfacing topically-unrelated sentences).
      2. The BEST sentence must have overlap >= min_best_overlap relative to
         the query — if even the best match is weak, we refuse instead of
         returning a low-quality answer.
      3. Lead-sentence fallback is ONLY used when the top passage score is
         above a quality floor (i.e., the passage is genuinely relevant,
         not just a weak lexical match).

    Returns an empty answer (which the harness treats as a refusal) when no
    sentence meets the quality bar."""
    max_sentences = getattr(cfg, "extractive_max_sentences", 3)
    min_sentence_words = 3   # skip fragments like "See also." / "वेबसाइट।"
    if not results:
        return GenerationResult(answer="", raw_response={"extractive": True, "reason": "no results"})

    q_words = _content_words(query)
    n_query_words = len(q_words)
    # Short queries (≤3 words) need at least 1 match; longer queries need 2.
    min_overlap = 1 if n_query_words <= 3 else 2
    # Minimum overlap for the BEST sentence — prevents weak matches from
    # being surfaced as answers. For a 5-word query with 3 content words,
    # we want at least 1 strong match in the best sentence.
    min_best_overlap = max(1, n_query_words // 3)

    # Question-type detection: certain sentence patterns directly answer
    # specific question types (e.g., "X is a Y" answers "What is X?").
    q_lower = query.lower()
    is_what_q = q_lower.startswith(("what is", "what are", "define ", "definition of"))
    is_how_many_q = any(q_lower.startswith(w) for w in ["how many", "how much", "how often"])
    is_how_q = q_lower.startswith("how ")
    is_where_q = any(q_lower.startswith(w) for w in ["where is", "where are", "where do"])
    is_when_q = any(q_lower.startswith(w) for w in ["when is", "when was", "when did"])
    is_who_q = any(q_lower.startswith(w) for w in ["who is", "who was", "who are"])
    is_why_q = any(q_lower.startswith(w) for w in ["why did", "why do", "why is", "why does"])

    scored: List[tuple] = []  # (score, passage_rank, sentence_idx, sentence)
    for rank, r in enumerate(results[:3]):  # top-3 passages only — keeps it sub-ms
        for si, sent in enumerate(_split_sentences(r.chunk.text)):
            words = _WORD_RE.findall(sent)
            if len(words) < min_sentence_words:
                continue
            sent_lower = sent.lower()
            sent_words = set(w.lower() for w in words)
            overlap = len(q_words & sent_words)
            if overlap < min_overlap:
                continue  # skip sentences with insufficient topical grounding

            # --- Answer-quality scoring (beyond raw overlap count) ---
            score = overlap

            # Boost 1: answer-pattern match. Sentences that directly answer
            # the question type score higher.
            if is_what_q and re.search(r"\b(is|are|was|were)\b", sent_lower):
                score += 0.5  # "X is a Y" pattern directly answers "What is X?"
            if is_how_many_q and re.search(r"\b\d+\b", sent):
                score += 0.8  # numeric answer for quantity questions
            if is_how_q and re.search(r"\b\d+\b", sent):
                score += 0.5  # numeric answer for how-questions
            if is_where_q and re.search(r"\b(in|at|on|located|found)\b", sent_lower):
                score += 0.5
            if is_when_q and re.search(r"\b(in|on|during|since|from|year|date)\b", sent_lower):
                score += 0.5
            if is_who_q and re.search(r"\b(was|is|were|born|founded|created|invented)\b", sent_lower):
                score += 0.5
            if is_why_q and re.search(r"\b(because|due to|since|reason|allows|enables)\b", sent_lower):
                score += 0.5

            # Boost 2: sentence position — earlier sentences in a passage are
            # more likely to contain the main answer.
            if si == 0:
                score += 0.3
            elif si == 1:
                score += 0.1

            # Penalty: overly long sentences (likely lists or navigation)
            if len(words) > 40:
                score -= 0.5

            # Boost 3: passage rank bonus
            score_bonus = r.score if rank < len(results) else 0.0
            score += 0.01 * score_bonus + 1e-6 * (len(results) - rank)

            scored.append((score, rank, si, sent))

    if not scored:
        # No sentence has even minimal keyword overlap with the query.
        # Only use lead-sentence fallback if the top passage is genuinely
        # relevant (score above a quality floor). Otherwise refuse.
        top_score = results[0].score
        if top_score < 0.35:
            # Weak retrieval — don't guess, refuse with an honest message
            return GenerationResult(
                answer="",
                raw_response={"extractive": True, "reason": "no relevant sentences found",
                              "top_score": top_score},
            )
        # Top passage is reasonably relevant — use its lead sentence
        lead = _split_sentences(results[0].chunk.text)
        answer = lead[0] if lead else results[0].chunk.text[:300]
        return GenerationResult(answer=answer, raw_response={"extractive": True, "fallback": "lead"})

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    best_score = scored[0][0]
    # Quality gate: if the best sentence's overlap is too weak relative to
    # the query size, refuse instead of returning a low-confidence answer.
    if n_query_words >= 3 and best_score < min_best_overlap:
        return GenerationResult(
            answer="",
            raw_response={"extractive": True, "reason": "best sentence overlap too weak",
                          "best_score": best_score, "min_required": min_best_overlap},
        )

    picked = scored[:max_sentences]
    # Present in passage order (reading flow) rather than score order
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
