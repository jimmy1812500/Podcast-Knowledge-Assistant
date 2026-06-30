"""
Speaker diarization via pyannote.audio.

Identifies WHO is speaking at each timestamp. Returns a list of SpeakerSegment
objects that the pipeline overlaps with Whisper ASR output to label each
transcript chunk with the dominant speaker.

Requirements:
    1. pip install pyannote.audio
    2. Accept model license at https://huggingface.co/pyannote/speaker-diarization-3.1
    3. Set HF_TOKEN env var (https://huggingface.co/settings/tokens)

The pipeline model (~1 GB) is downloaded once and cached in
~/.cache/huggingface/hub/ on first use.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"

# Singleton — loading the pipeline takes ~10s and ~1 GB RAM.
_pipeline = None


@dataclass
class SpeakerSegment:
    start: float   # seconds
    end: float
    speaker: str   # "SPEAKER_00", "SPEAKER_01", …


def _load_pipeline(hf_token: str):
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline
        _pipeline = Pipeline.from_pretrained(DIARIZE_MODEL, token=hf_token)
    return _pipeline


def _diarize_sync(audio_path: Path, hf_token: str) -> list[SpeakerSegment]:
    pipeline = _load_pipeline(hf_token)
    annotation = pipeline(str(audio_path))
    return [
        SpeakerSegment(start=seg.start, end=seg.end, speaker=label)
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]


async def diarize(audio_path: Path, hf_token: str | None = None) -> list[SpeakerSegment]:
    """
    Run pyannote speaker diarization on an audio file.

    Returns SpeakerSegments sorted by start time. Runs in a thread executor
    so it doesn't block the event loop during inference.

    Raises RuntimeError if HF_TOKEN is not available.
    """
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for pyannote diarization.\n"
            "  1. Accept the model license at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  2. export HF_TOKEN=hf_..."
        )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _diarize_sync, audio_path, token)


def dominant_speaker(
    time_start: float,
    time_end: float,
    speaker_segs: list[SpeakerSegment],
) -> str:
    """
    Return the speaker with the most cumulative overlap in [time_start, time_end].
    Returns "UNKNOWN" when no diarization segment covers the window.
    """
    tally: dict[str, float] = {}
    for s in speaker_segs:
        overlap = min(time_end, s.end) - max(time_start, s.start)
        if overlap > 0:
            tally[s.speaker] = tally.get(s.speaker, 0.0) + overlap
    return max(tally, key=tally.__getitem__) if tally else "UNKNOWN"
