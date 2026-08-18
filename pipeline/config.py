"""
Central configuration for the voice-RAG pipeline.
All tunables live here so the coding agent (or you) can swap providers,
chunk sizes, and thresholds without hunting through the codebase.
"""
import os
from dataclasses import dataclass, field


@dataclass
class STTConfig:
    # "auto" -> Sarvam STT when SARVAM_API_KEY is set (task-spec compliant,
    # Indic-focused), else Groq Whisper (free tier, hosted, works on
    # serverless), else local Whisper fallback. Other values: "sarvam" |
    # "groq" | "whisper" | "elevenlabs".
    provider: str = "auto"
    # Groq's OpenAI-compatible Whisper endpoint — reuses GROQ_API_KEY (same
    # key as generation, no separate account). Verified live Aug 2026.
    # Kept as fallback when no Sarvam key is available.
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_stt_endpoint: str = "https://api.groq.com/openai/v1/audio/transcriptions"
    groq_stt_model: str = "whisper-large-v3-turbo"  # v3-turbo: best price/perf; whisper-large-v3 for max accuracy
    sarvam_api_key: str = field(default_factory=lambda: os.getenv("SARVAM_API_KEY", ""))
    sarvam_endpoint: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_model: str = "saaras:v3"   # current Sarvam STT model (verified against live docs)
    sarvam_mode: str = "transcribe"    # transcribe | translate | verbatim | translit | codemix
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    whisper_model: str = ""  # empty -> auto-resolve (repo-local small model, HF cache, or base id)
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    language_code: str = "hi-IN"  # MSMARCO-XI is Hindi; override per query if needed
    timeout_s: float = 8.0
    max_retries: int = 2


@dataclass
class ChunkingConfig:
    # Multiple strategies are implemented in pipeline/chunking.py.
    # `active_strategy` picks which one the harness uses by default;
    # all strategies can also be run side-by-side for comparison (see benchmark/).
    active_strategy: str = "metadata_aware"  # "fixed" | "sentence" | "metadata_aware" | "hybrid"
    fixed_chunk_size: int = 200          # characters
    fixed_chunk_overlap: int = 40        # characters
    sentence_window: int = 3             # sentences per chunk
    sentence_overlap: int = 1            # sentences of overlap between windows


@dataclass
class RetrievalConfig:
    top_k: int = 5
    hybrid_alpha: float = 0.5   # weight between BM25 (lexical) and TF-IDF cosine (semantic-ish), 0=pure BM25, 1=pure TFIDF
    min_relevance_score: float = 0.08  # below this -> treated as "no good context found" (guardrail signal)


@dataclass
class GenerationConfig:
    # "auto" -> Groq when GROQ_API_KEY is set, else Anthropic when
    # ANTHROPIC_API_KEY is set, else a fast-fail GenerationError. Other
    # values: "groq" | "anthropic" (mirrors the STT provider pattern).
    provider: str = "auto"
    # --- Anthropic ---
    model: str = "claude-sonnet-4-6"
    api_url: str = "https://api.anthropic.com/v1/messages"
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    # --- Groq (OpenAI-compatible chat completions) ---
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_endpoint: str = "https://api.groq.com/openai/v1/chat/completions"
    # Fastest model in Groq's CURRENT catalog (Aug 2026) — the llama models
    # (llama-3.3-70b-versatile / 3.1-8b-instant) were removed from the API
    # (404) and measured gpt-oss-20b < gpt-oss-120b < qwen3.6-27b (qwen also
    # wastes tokens on <think> blocks unless reasoning is disabled).
    groq_model: str = "openai/gpt-oss-20b"
    max_tokens: int = 512
    timeout_s: float = 15.0
    max_retries: int = 2
    use_mock: bool = False  # dev/testing only — see generation.generate_answer_mock(). Must be False for submission runs.


@dataclass
class GuardrailConfig:
    # Off-topic / no-relevant-context detection. Tuned against the REAL
    # corpus (see DECISIONS.md): min-max normalized scores are always ~1.0
    # on a general-domain corpus, so the meaningful signals are the RAW
    # top-hit TF-IDF cosine and content-word overlap between the query and
    # the top-3 retrieved chunks. A query is refused only when BOTH are
    # below their floors (avoids false refusals of genuine corpus queries —
    # measured 0% false refusals on 120 real gold queries with these values).
    off_topic_similarity_floor: float = 0.35     # raw top-hit TF-IDF cosine floor
    off_topic_overlap_floor: float = 0.40        # content-word overlap (top-3 context) floor
    # Kept as a legacy safety net for corpora where normalized scores still
    # carry signal (e.g. tiny toy corpora); on the real corpus it rarely fires.
    off_topic_normalized_floor: float = 0.05
    grounding_overlap_floor: float = 0.15        # min lexical overlap between answer and retrieved context
    # Corpus-driven stopwords (populated by the harness from the indexed
    # chunks): tokens common enough to be uninformative are stripped from the
    # query before the off-topic overlap/cosine signals are computed. Falls
    # back to a hand-curated multilingual list when not populated.
    stopwords: frozenset = frozenset()
    unsafe_keywords: tuple = (
        # English
        "bomb", "explosive", "weapon", "kill", "suicide", "self-harm",
        "hack into", "malware", "child abuse",
        # Hindi (the demo's primary language)
        "बम", "विस्फोटक", "हथियार", "हत्या", "आत्महत्या", "आत्म-हत्या",
        "मारने", "मारना", "मैलवेयर", "हैक", "बच्चों का शोषण",
    )
    refusal_message: str = (
        "I don't have enough grounded context to answer that reliably. "
        "Could you rephrase, or ask something covered by the dataset?"
    )


@dataclass
class PipelineConfig:
    stt: STTConfig = field(default_factory=STTConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    latency_budget_ms: float = 200.0
