"""
Whisper ASR interface — transcribes audio to structured text + segments.

Uses faster-whisper (CTranslate2 backend) for low-memory, CPU-friendly inference.
Model weights are downloaded on first use to ~/.cache/huggingface/hub/.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = "base"  # tiny | base | small | medium | large-v3

# compute_type per device: int8 is optimal on CPU, float16 on CUDA
_COMPUTE_TYPE: dict[str, str] = {
    "cpu": "int8",
    "cuda": "float16",
}


@dataclass
class Segment:
    start: float  # seconds from audio start
    end: float
    text: str
    confidence: float  # [0.0, 1.0] derived from avg_logprob


@dataclass
class Transcript:
    text: str
    segments: list[Segment]
    language: str


_model_cache: dict[tuple[str, str], WhisperModel] = {}


def _load_model(size: str, device: str) -> WhisperModel:
    key = (size, device)
    if key not in _model_cache:
        compute_type = _COMPUTE_TYPE.get(device, "int8")
        _model_cache[key] = WhisperModel(size, device=device, compute_type=compute_type)
    return _model_cache[key]


def _transcribe_sync(audio_path: Path, model_size: str, device: str) -> Transcript:
    model = _load_model(model_size, device)
    raw_segments, info = model.transcribe(  # type: ignore[arg-type]
        str(audio_path),
        beam_size=5,
        vad_filter=True,  # skip silent regions
    )

    segments: list[Segment] = []
    texts: list[str] = []
    for seg in raw_segments:
        # avg_logprob ∈ (-∞, 0]; exp maps it to (0, 1]
        confidence = min(1.0, math.exp(max(-5.0, seg.avg_logprob)))
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(start=seg.start, end=seg.end, text=text, confidence=confidence))
        texts.append(text)

    return Transcript(
        text=" ".join(texts),
        segments=segments,
        language=info.language,
    )


async def transcribe(
    audio_path: Path,
    model_size: str = WHISPER_MODEL_SIZE,
    device: str = "cpu",
) -> Transcript:
    """Transcribe an audio file, returning full text + timestamped segments."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe_sync, audio_path, model_size, device)
