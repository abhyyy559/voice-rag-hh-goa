"""
Guardrails run at three points in the harness:
  1. PRE-retrieval: unsafe-input screen on the raw transcript.
  2. POST-retrieval, PRE-generation: off-topic / no-relevant-context screen
     using the raw top-hit TF-IDF cosine and content-word overlap between
     the query and the retrieved chunks (if nothing relevant was retrieved,
     don't generate — refuse instead).
  3. POST-generation: grounding check — does the answer actually overlap
     with the retrieved context, or did the model wander off and hallucinate?

Each check returns a GuardrailVerdict so the harness can log *why* it
refused, not just that it refused — that's the "knows when not to answer"
requirement.

Tuning note (see DECISIONS.md): on a general-domain corpus like MS MARCO,
min-max normalized retrieval scores are ~1.0 for almost every query (the
best hit always normalizes to 1), so they carry no off-topic signal. The
working signals are (a) the RAW TF-IDF cosine of the top hit and (b) the
fraction of the query's content words (stopwords stripped) that appear in
the top-3 retrieved chunks. A query is refused only when BOTH are below
floor — measured 0% false refusals on 120 real gold queries with the
default floors.
"""
import re
from dataclasses import dataclass
from typing import List

from .config import GuardrailConfig
from .retrieval import RetrievalResult


@dataclass
class GuardrailVerdict:
    passed: bool
    reason: str = ""
    stage: str = ""


def check_unsafe_input(query: str, cfg: GuardrailConfig) -> GuardrailVerdict:
    q_lower = query.lower()
    for kw in cfg.unsafe_keywords:
        if kw in q_lower:
            return GuardrailVerdict(passed=False, reason=f"matched unsafe keyword: '{kw}'", stage="unsafe_input")
    return GuardrailVerdict(passed=True, stage="unsafe_input")


# Stopwords stripped from the query before computing content-word overlap,
# so generic question words can't satisfy the relevance check by themselves.
# Covers Hindi (Devanagari) and the English question/function words that
# appear in code-mixed queries.
_STOPWORDS = frozenset("""
क्या है हैं कैसे में की का के को से पर और कौन कौनसा कौन सा कितना कितने कितनी
होता होती होते करता करती करते नहीं जाता जाती जाते लिए किस किसी वाले वाला वाली
कीजिए कैसा सा सी ही भी तो जो यह वह इस उस एक और हो हुआ हुए था थी थे करना करने
कर सकता सकते सकती रहता रहती रहते आदि क्योंकि अगर तब जब तक कि जिस वही यही उसे
इसे उन्होंने मैं तुम हम आप क्या होता क्या होती क्या करते किसका किसकी किसके
कहाँ कब क्यों किसलिए कैसा कैसी कौनसे कौनसी बताइए बताओ पाया पाई पाए पाई जाती
हुए करें कर सकेंगे हुई हुआ जाना जाने लगता लगती होने रखता रखती बनता बनती देना
देता देती लेना लेता लेती करनी करने के लिए वगैरह आज कल

the a an of to in is are was were do does did what how which who why when where
for with on at by from it its and or not can could would should will this that
these those i you we they he she be been being have has had their there here
about into over under than then them
tell give show make get go come take know see find use need try keep let say said
put run turn move play live feel believe think become leave follow stop help start
work call seem ask look long much many also very well back still most even after
here there only now again just because way may down been before being between both
each same another such every own rather quite really already since while during
until against among throughout within without before behind below above across
along around beyond through during except inside outside above below
""".split())


_TOKEN_RE = re.compile(r"[a-zA-Z\u0900-\u097F\u0980-\u09FF\u0C00-\u0C7F]+")  # Latin + Devanagari + Bengali + Telugu


def _content_words(text: str, stopwords: frozenset = frozenset()) -> set:
    # Corpus-driven stopwords (high-frequency tokens) UNION the hand-curated
    # base list: the base covers question/function words that rarely appear in
    # passages (so corpus frequency misses them), the corpus list covers
    # domain function words the hand list misses.
    stops = _STOPWORDS | frozenset(stopwords)
    return set(w for w in _TOKEN_RE.findall(text.lower()) if w not in stops and len(w) >= 2)


def _query_content_overlap(query: str, results: List[RetrievalResult],
                           stopwords: frozenset = frozenset()) -> float:
    """Fraction of the query's content words present in the top-3 chunks."""
    q_words = _content_words(query, stopwords)
    if not q_words:
        return 0.0
    ctx_words = set()
    for r in results[:3]:
        ctx_words |= _content_words(r.chunk.text, stopwords)
    return len(q_words & ctx_words) / len(q_words)


def check_off_topic(query: str, results: List[RetrievalResult], cfg: GuardrailConfig) -> GuardrailVerdict:
    if not results:
        return GuardrailVerdict(
            passed=False, reason="retrieval returned no results — corpus has nothing for this query",
            stage="off_topic",
        )
    top = results[0]
    # Legacy safety net for toy corpora where the normalized score still means
    # something; on the real corpus normalized scores are ~1.0 and this won't
    # fire (that's expected — see module docstring).
    if top.score < cfg.off_topic_normalized_floor:
        return GuardrailVerdict(
            passed=False,
            reason=f"top retrieval score {top.score:.3f} below floor {cfg.off_topic_normalized_floor} — "
                   f"likely off-topic or not covered by the dataset",
            stage="off_topic",
        )
    overlap = _query_content_overlap(query, results, cfg.stopwords)
    raw_tfidf = top.tfidf_score
    q_words = _content_words(query, cfg.stopwords)

    # HARD RULE: if the query has content words but NONE of them appear in
    # the retrieved chunks, the results are topically unrelated regardless
    # of TF-IDF score. This catches the common case where English function
    # words ("the", "is", "what") inflate TF-IDF cosine against English
    # content in a multilingual corpus, while the actual query topic is
    # absent from the results.
    if q_words and overlap == 0.0:
        return GuardrailVerdict(
            passed=False,
            reason=f"zero content-word overlap ({len(q_words)} query words, 0 found in context) — "
                   f"results are topically unrelated to the query",
            stage="off_topic",
        )

    # Standard floor check: refuse when BOTH signals are below threshold.
    if raw_tfidf < cfg.off_topic_similarity_floor and overlap < cfg.off_topic_overlap_floor:
        return GuardrailVerdict(
            passed=False,
            reason=f"no relevant context: top-hit raw TF-IDF {raw_tfidf:.3f} below floor "
                   f"{cfg.off_topic_similarity_floor} AND query content-word overlap "
                   f"{overlap:.2f} below floor {cfg.off_topic_overlap_floor} — "
                   f"query not covered by the dataset",
            stage="off_topic",
        )
    return GuardrailVerdict(passed=True, stage="off_topic")


def check_grounding(answer: str, results: List[RetrievalResult], cfg: GuardrailConfig) -> GuardrailVerdict:
    """Lexical-overlap grounding check: cheap, fast (<1ms), no extra model
    call, and catches the common case of the model answering from its own
    knowledge instead of the retrieved passages. The coding agent can
    upgrade this to an NLI/entailment model or an LLM self-critique pass
    once latency headroom and network access allow it."""
    context_text = " ".join(r.chunk.text for r in results)
    context_words = _content_words(context_text)
    answer_words = _content_words(answer)
    if not answer_words:
        return GuardrailVerdict(passed=False, reason="empty or unparseable answer", stage="grounding")
    overlap = len(answer_words & context_words) / len(answer_words)
    if overlap < cfg.grounding_overlap_floor:
        return GuardrailVerdict(
            passed=False,
            reason=f"answer/context word overlap {overlap:.2f} below floor {cfg.grounding_overlap_floor} — "
                   f"answer may not be grounded in retrieved context",
            stage="grounding",
        )
    return GuardrailVerdict(passed=True, stage="grounding")
