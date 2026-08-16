"""
Answer generation stage. Uses the Anthropic Messages API with a strict
grounding instruction so the model prefers "not enough information" over
guessing — the guardrails.check_grounding step then double-checks that
behaviorally instead of just trusting the prompt.
"""
import os
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4),
       retry=retry_if_exception_type(GenerationError), reraise=True)
def _generate_answer_once(query: str, results: List[RetrievalResult], cfg: GenerationConfig,
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


def generate_answer(query: str, results: List[RetrievalResult], cfg: GenerationConfig) -> GenerationResult:
    """Missing-key errors are checked OUTSIDE the retried inner call so a
    missing key fails fast (single attempt) instead of burning 3 retries."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise GenerationError("ANTHROPIC_API_KEY not set in environment")
    prompt = _build_prompt(query, results)
    return _generate_answer_once(query, results, cfg, api_key, prompt)
