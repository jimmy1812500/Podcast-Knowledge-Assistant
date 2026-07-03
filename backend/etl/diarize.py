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

# Singleton keyed by device — loading the pipeline takes ~10s and ~1 GB RAM.
_pipeline_cache: dict[str, object] = {}


@dataclass
class SpeakerSegment:
    start: float  # seconds
    end: float
    speaker: str  # "SPEAKER_00", "SPEAKER_01", …


def _load_pipeline(hf_token: str, device: str):
    if device not in _pipeline_cache:
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(DIARIZE_MODEL, token=hf_token)
        pipeline.to(torch.device(device))
        _pipeline_cache[device] = pipeline
    return _pipeline_cache[device]


def _diarize_sync(audio_path: Path, hf_token: str, device: str) -> list[SpeakerSegment]:
    pipeline = _load_pipeline(hf_token, device)
    output = pipeline(str(audio_path))
    # Unwrap the annotation object — API differs across pyannote versions:
    #   old (≤3.1): pipeline() returns Annotation directly (has itertracks)
    #   new (≥3.2): pipeline() returns DiarizeOutput with a .diarization field
    if hasattr(output, "itertracks"):
        annotation = output
    elif hasattr(output, "speaker_diarization"):
        annotation = output.speaker_diarization
    elif hasattr(output, "diarization"):
        annotation = output.diarization
    elif hasattr(output, "annotation"):
        annotation = output.annotation
    else:
        public_attrs = [a for a in dir(output) if not a.startswith("_")]
        raise AttributeError(
            f"Cannot extract Annotation from {type(output).__name__}. "
            f"Available attributes: {public_attrs}"
        )
    return [
        SpeakerSegment(start=seg.start, end=seg.end, speaker=label)
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]


async def diarize(
    audio_path: Path,
    hf_token: str | None = None,
    device: str = "cpu",
) -> list[SpeakerSegment]:
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
    return await loop.run_in_executor(None, _diarize_sync, audio_path, token, device)


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
