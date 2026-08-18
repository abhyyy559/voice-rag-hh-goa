"""
Speech-to-text stage.

Providers (config.py -> STTConfig.provider):
  - "sarvam"    : Sarvam AI REST API (Indic-focused) — PRIMARY per task spec.
                  Endpoint POST https://api.sarvam.ai/speech-to-text, header
                  api-subscription-key, multipart file/language_code/model
                  (saaras:v3)/mode, response {transcript}.
                  Requires SARVAM_API_KEY.
  - "groq"      : Groq's OpenAI-compatible Whisper endpoint (whisper-large-v3
                  / v3-turbo) — FALLBACK. Reuses GROQ_API_KEY (same key as
                  generation), free tier, hosted. VERIFIED live Aug 2026.
  - "whisper"   : local faster-whisper (no API key, fully offline). Zero-key
                  fallback for dev runs.
  - "elevenlabs": ElevenLabs Scribe v1. One-line config swap.
  - "auto"      : (default) Sarvam when SARVAM_API_KEY is set, else Groq when
                  GROQ_API_KEY is set, else local Whisper.
"""
import io
import os
import requests
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import STTConfig


class STTError(Exception):
    pass


@dataclass
class TranscriptionResult:
    text: str
    language_code: str
    provider: str
    raw_response: dict


def resolve_stt_provider(cfg: STTConfig) -> str:
    if cfg.provider == "auto":
        if cfg.sarvam_api_key:
            return "sarvam"
        if cfg.groq_api_key:
            return "groq"
        return "whisper"
    return cfg.provider


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4),
       retry=retry_if_exception_type(STTError), reraise=True)
def _transcribe_sarvam_once(audio_bytes: bytes, filename: str, cfg: STTConfig) -> TranscriptionResult:
    try:
        resp = requests.post(
            cfg.sarvam_endpoint,
            headers={"api-subscription-key": cfg.sarvam_api_key},
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={"language_code": cfg.language_code,
                  "model": cfg.sarvam_model,
                  "mode": cfg.sarvam_mode},
            timeout=cfg.timeout_s,
        )
        if resp.status_code in (401, 403):
            raise STTError(f"Sarvam auth failed ({resp.status_code}) — check SARVAM_API_KEY")
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise STTError(str(e)) from e

    text = (data.get("transcript") or "").strip()
    if not text:
        raise STTError(f"empty transcript from Sarvam (raw: {str(data)[:200]})")
    return TranscriptionResult(text=text, language_code=cfg.language_code,
                               provider="sarvam", raw_response=data)


def transcribe_sarvam(audio_bytes: bytes, filename: str, cfg: STTConfig) -> TranscriptionResult:
    """Key check outside the retried call: a missing key fails fast instead
    of burning 3 retries with backoff."""
    if not cfg.sarvam_api_key:
        raise STTError("SARVAM_API_KEY not set")
    return _transcribe_sarvam_once(audio_bytes, filename, cfg)


def _whisper_model_path() -> str:
    """Resolve a locally available faster-whisper model (repo-local dir or
    HF cache snapshot), else return the HF repo id so faster-whisper
    downloads it on first use."""
    repo_local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "models", "faster-whisper-small")
    if os.path.exists(os.path.join(repo_local, "model.bin")):
        return repo_local
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    for d in os.listdir(hub) if os.path.isdir(hub) else []:
        if d.startswith("models--") and ("whisper" in d.lower()):
            snap = os.path.join(hub, d, "snapshots")
            if os.path.isdir(snap):
                snaps = os.listdir(snap)
                if snaps:
                    p = os.path.join(snap, snaps[0])
                    if os.path.exists(os.path.join(p, "model.bin")):
                        return p
    return "Systran/faster-whisper-base"


_whisper_model = None
_whisper_model_id = None


def transcribe_whisper(audio_bytes: bytes, filename: str, cfg: STTConfig) -> TranscriptionResult:
    global _whisper_model, _whisper_model_id
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise STTError("faster-whisper not installed — run: pip install faster-whisper "
                       "(or set SARVAM_API_KEY to use Sarvam)") from e

    model_ref = cfg.whisper_model or _whisper_model_path()
    if _whisper_model is None or _whisper_model_id != model_ref:
        _whisper_model = WhisperModel(model_ref, device=cfg.whisper_device,
                                      compute_type=cfg.whisper_compute_type)
        _whisper_model_id = model_ref

    # faster-whisper decodes via PyAV; pass a file-like object (raw bytes
    # aren't accepted). Works for mp3/wav/webm/opus. The language is forced
    # from config ("hi") — the tiny base model otherwise drifts into
    # transliteration or the wrong script on Indic audio.
    lang = cfg.language_code.split("-")[0].lower() or None
    segments, info = _whisper_model.transcribe(
        io.BytesIO(audio_bytes), language=lang, vad_filter=True,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    if not text:
        raise STTError("local whisper returned empty transcript")
    lang = info.language or cfg.language_code
    return TranscriptionResult(text=text, language_code=lang,
                               provider="whisper", raw_response={"model": model_ref, "lang": lang})


class RateLimitedSTTError(STTError):
    """STT API returned 429 (rate limit) — retry with a longer backoff."""

    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


def _stt_wait(retry_state) -> float:
    exc = retry_state.outcome.exception()
    if isinstance(exc, RateLimitedSTTError) and exc.retry_after > 0:
        return max(exc.retry_after, 4.0)
    return min(2 ** (retry_state.attempt_number - 1), 10.0)


@retry(stop=stop_after_attempt(4), wait=_stt_wait,
       retry=retry_if_exception_type(STTError), reraise=True)
def _transcribe_groq_once(audio_bytes: bytes, filename: str, cfg: STTConfig) -> TranscriptionResult:
    """Groq Whisper transcription (OpenAI-compatible). Language is ISO-639-1
    ("hi"), derived from the config's "hi-IN"."""
    lang = cfg.language_code.split("-")[0].lower() or "hi"
    try:
        resp = requests.post(
            cfg.groq_stt_endpoint,
            headers={"Authorization": f"Bearer {cfg.groq_api_key}"},
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={"model": cfg.groq_stt_model, "language": lang, "response_format": "json"},
            timeout=cfg.timeout_s,
        )
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "0") or 0)
            raise RateLimitedSTTError(f"Groq STT rate limit (429), retry after {retry_after}s",
                                      retry_after=retry_after)
        if resp.status_code in (401, 403):
            raise STTError(f"Groq STT auth failed ({resp.status_code}) — check GROQ_API_KEY")
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise STTError(str(e)) from e

    text = (data.get("text") or "").strip()
    if not text:
        raise STTError(f"empty transcript from Groq STT (raw: {str(data)[:200]})")
    return TranscriptionResult(text=text, language_code=lang, provider="groq", raw_response=data)


def transcribe_groq(audio_bytes: bytes, filename: str, cfg: STTConfig) -> TranscriptionResult:
    """Key check outside the retried call: a missing key fails fast instead
    of burning 4 retries with backoff."""
    if not cfg.groq_api_key:
        raise STTError("GROQ_API_KEY not set")
    return _transcribe_groq_once(audio_bytes, filename, cfg)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4),
       retry=retry_if_exception_type(STTError), reraise=True)
def transcribe_elevenlabs(audio_bytes: bytes, filename: str, cfg: STTConfig) -> TranscriptionResult:
    if not cfg.elevenlabs_api_key:
        raise STTError("ELEVENLABS_API_KEY not set")
    try:
        resp = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": cfg.elevenlabs_api_key},
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={"model_id": "scribe_v1"},
            timeout=cfg.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise STTError(str(e)) from e

    text = (data.get("text") or "").strip()
    if not text:
        raise STTError("empty transcript from ElevenLabs")
    return TranscriptionResult(text=text, language_code=cfg.language_code, provider="elevenlabs", raw_response=data)


def transcribe(audio_bytes: bytes, filename: str, cfg: STTConfig) -> TranscriptionResult:
    provider = resolve_stt_provider(cfg)
    if provider == "groq":
        return transcribe_groq(audio_bytes, filename, cfg)
    elif provider == "whisper":
        return transcribe_whisper(audio_bytes, filename, cfg)
    elif provider == "sarvam":
        return transcribe_sarvam(audio_bytes, filename, cfg)
    elif provider == "elevenlabs":
        return transcribe_elevenlabs(audio_bytes, filename, cfg)
    raise ValueError(f"unknown STT provider: {provider}")
