"""
Speech-to-text stage.

Providers (config.py -> STTConfig.provider):
  - "sarvam"    : Sarvam AI REST API (Indic-focused, fits MSMARCO-XI).
                  Request/response shape VERIFIED against live docs
                  (https://docs.sarvam.ai/api-reference/speech-to-text/transcribe)
                  and a live unauthenticated probe (Aug 2026): endpoint
                  POST https://api.sarvam.ai/speech-to-text, header
                  api-subscription-key, multipart fields file/language_code/
                  model/mode; response {request_id, transcript, language_code}.
                  Current model is saaras:v3 (saarika:v2 is no longer listed).
  - "whisper"   : local faster-whisper (no API key, fully offline). Uses a
                  cached HF model if present. Useful for demo/dev runs when no
                  key is available; NOT the configured production path.
  - "elevenlabs": stub kept so the provider remains a one-line config swap.
  - "auto"      : (default) Sarvam when SARVAM_API_KEY is set, else local
                  Whisper — keeps the pipeline functional without a key while
                  preferring the production provider.
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
        return "sarvam" if cfg.sarvam_api_key else "whisper"
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
    if provider == "sarvam":
        return transcribe_sarvam(audio_bytes, filename, cfg)
    elif provider == "whisper":
        return transcribe_whisper(audio_bytes, filename, cfg)
    elif provider == "elevenlabs":
        return transcribe_elevenlabs(audio_bytes, filename, cfg)
    raise ValueError(f"unknown STT provider: {provider}")
