"""
ETL pipeline orchestrator: audio file → ASR + diarization → split → embed → ChromaDB.

Shared by both /upload (direct file upload) and /ingest/podcast (RSS download).

Diarization is optional. When HF_TOKEN is set, pyannote runs in parallel with
Whisper and the dominant speaker label is stored as chunk metadata. When
HF_TOKEN is absent, speaker defaults to "UNKNOWN" and no diarization is attempted.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

from etl.asr import Transcript, transcribe
from etl.diarize import SpeakerSegment, diarize, dominant_speaker
from etl.embeddings import store_chunks
from etl.splitter import Chunk, split_text


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
    # The +1 mirrors the " ".join(" ") separator used in _transcribe_sync.
    pos = 0
    seg_spans: list[tuple[int, int, float, float, float]] = []
    for seg in transcript.segments:
        seg_end = pos + len(seg.text) + 1
        seg_spans.append((pos, seg_end, seg.start, seg.end, seg.confidence))
        pos = seg_end

    result: list[dict[str, Any]] = []
    for chunk in chunks:
        overlapping = [
            s for s in seg_spans
            if s[0] < chunk.char_end and s[1] > chunk.char_start
        ]
        if overlapping:
            t_start = min(s[2] for s in overlapping)
            t_end   = max(s[3] for s in overlapping)
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
                "timestamp_end":   round(t_end, 2),
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
) -> dict[str, Any]:
    """
    Full ETL pipeline on a single audio file.

    Steps:
        1. Whisper ASR + pyannote diarization (run in parallel)
        2. Split transcript into overlapping chunks
        3. Map each chunk → timestamps + dominant speaker
        4. Embed with Gemini text-embedding-004 + upsert to ChromaDB

    Diarization is skipped gracefully when HF_TOKEN is not set; speaker
    metadata will be "UNKNOWN" for all chunks in that case.

    Returns summary dict: source, language, segments, chunks_stored, speakers_found
    """
    source = source_name or audio_path.name

    # Run sequentially — both are CPU-bound (CTranslate2 + PyTorch) and
    # compete for cores when parallel, making each slower. Whisper first,
    # diarization after so it doesn't steal CPU during transcription.
    transcript   = await transcribe(audio_path, model_size=whisper_model)
    speaker_segs = await _safe_diarize(audio_path)

    chunks     = split_text(transcript.text, chunk_size=chunk_size, overlap=overlap)
    chunk_meta = _build_chunk_meta(chunks, transcript, source, speaker_segs)
    stored     = await store_chunks(chunk_meta)

    speakers_found = sorted({c["speaker"] for c in chunk_meta} - {"UNKNOWN"})

    return {
        "source": source,
        "language": transcript.language,
        "segments": len(transcript.segments),
        "chunks_stored": stored,
        "speakers_found": speakers_found,
    }


async def _safe_diarize(audio_path: Path) -> list[SpeakerSegment]:
    """Run diarization if HF_TOKEN is set; return [] silently otherwise."""
    if not os.environ.get("HF_TOKEN"):
        return []
    try:
        return await diarize(audio_path)
    except Exception as exc:
        print(f"  [diarize] WARNING: {exc} — continuing without speaker labels")
        return []
