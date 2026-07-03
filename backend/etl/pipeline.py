"""
ETL pipeline orchestrator: audio file → ASR + diarization → split → embed → ChromaDB.

Shared by both /upload (direct file upload) and /ingest/podcast (RSS download).

Diarization is optional. Pass diarize=True to run_etl (and set HF_TOKEN in env)
to label each chunk with the dominant speaker. When diarize=False or HF_TOKEN
is absent, speaker defaults to "UNKNOWN".
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from backend.etl.asr import Transcript, transcribe
from backend.etl.diarize import SpeakerSegment, dominant_speaker
from backend.etl.diarize import diarize as _run_diarize
from backend.etl.embeddings import store_chunks
from backend.etl.splitter import Chunk, split_text


def _build_chunk_meta(
    chunks: list[Chunk],
    transcript: Transcript,
    source: str,
    speaker_segs: list[SpeakerSegment],
) -> list[dict[str, Any]]:
    """
    Build ChromaDB-ready dicts by mapping each chunk's character range to:
      - Whisper segment timestamps  (t_start, t_end, confidence)
      - pyannote dominant speaker   (speaker label covering most of the chunk)
    """
    # Build (char_start, char_end, time_start, time_end, confidence) per Whisper segment.
    # Advance pos by len+1 after each segment (the " " separator in " ".join),
    # but the span itself ends at pos+len so it doesn't bleed into the separator.
    pos = 0
    seg_spans: list[tuple[int, int, float, float, float]] = []
    for seg in transcript.segments:
        seg_end = pos + len(seg.text)
        seg_spans.append((pos, seg_end, seg.start, seg.end, seg.confidence))
        pos = seg_end + 1  # +1 for the " " separator between segments

    result: list[dict[str, Any]] = []
    for chunk in chunks:
        overlapping = [s for s in seg_spans if s[0] < chunk.char_end and s[1] > chunk.char_start]
        if overlapping:
            t_start = min(s[2] for s in overlapping)
            t_end = max(s[3] for s in overlapping)
            confidence = sum(s[4] for s in overlapping) / len(overlapping)
        else:
            t_start = t_end = 0.0
            confidence = 0.0

        speaker = dominant_speaker(t_start, t_end, speaker_segs)

        chunk_id = hashlib.sha1(f"{source}:{chunk.char_start}".encode()).hexdigest()
        result.append(
            {
                "chunk_id": chunk_id,
                "text": chunk.text,
                "source_file": source,
                "timestamp_start": round(t_start, 2),
                "timestamp_end": round(t_end, 2),
                "confidence_score": round(confidence, 4),
                "speaker": speaker,
            }
        )
    return result


async def run_etl(
    audio_path: Path,
    source_name: str | None = None,
    whisper_model: str = "base",
    chunk_size: int = 500,
    overlap: int = 100,
    device: str = "cpu",
    diarize: bool = False,
) -> dict[str, Any]:
    """
    Full ETL pipeline on a single audio file.

    Steps:
        1. Whisper ASR + pyannote diarization
        2. Split transcript into overlapping chunks
        3. Map each chunk → timestamps + dominant speaker
        4. Embed + upsert to ChromaDB

    device: "cpu" | "cuda" — controls both Whisper and pyannote inference device.
    diarize: explicitly enable speaker diarization (requires HF_TOKEN in env).

    Returns summary dict: source, language, segments, chunks_stored, speakers_found
    """
    source = source_name or audio_path.name

    # Run sequentially — both are CPU-bound (CTranslate2 + PyTorch) and
    # compete for cores when parallel, making each slower. Whisper first,
    # diarization after so it doesn't steal CPU during transcription.
    transcript = await transcribe(audio_path, model_size=whisper_model, device=device)
    speaker_segs = await _safe_diarize(audio_path, device=device, diarize=diarize)

    chunks = split_text(transcript.text, chunk_size=chunk_size, overlap=overlap)
    chunk_meta = _build_chunk_meta(chunks, transcript, source, speaker_segs)
    stored = await store_chunks(chunk_meta)

    speakers_found = sorted({c["speaker"] for c in chunk_meta} - {"UNKNOWN"})

    return {
        "source": source,
        "language": transcript.language,
        "segments": len(transcript.segments),
        "chunks_stored": stored,
        "speakers_found": speakers_found,
    }


async def _safe_diarize(
    audio_path: Path, device: str = "cpu", diarize: bool = False
) -> list[SpeakerSegment]:
    """Run diarization when requested and HF_TOKEN is set; return [] otherwise."""
    if not diarize:
        return []
    if not os.environ.get("HF_TOKEN"):
        print("  [diarize] SKIP — HF_TOKEN not set (add it to backend/.env)")
        return []
    try:
        segs = await _run_diarize(audio_path, device=device)
        print(f"  [diarize] OK — {len(segs)} speaker segments found")
        return segs
    except Exception as exc:
        print(f"  [diarize] ERROR: {exc} — continuing without speaker labels")
        return []
