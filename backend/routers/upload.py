"""
POST /upload — multipart audio file → ETL pipeline (ASR + embed + ChromaDB).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from backend.etl.pipeline import run_etl

router = APIRouter(tags=["upload"])

_ALLOWED_MIME = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/ogg",
    "audio/webm",
    "audio/flac",
    "audio/x-m4a",
}
_MAX_BYTES = 500 * 1024 * 1024  # 500 MB
_READ_CHUNK = 64 * 1024          # 64 KB streaming read


class UploadResult(BaseModel):
    source: str
    language: str
    segments: int
    chunks_stored: int


@router.post("/upload", response_model=UploadResult)
async def upload_audio(file: UploadFile) -> UploadResult:
    """
    Accept a WAV/MP3 audio file, transcribe it with Whisper, and store
    timestamped transcript embeddings in ChromaDB.

    Example:
        curl -X POST http://localhost:8000/upload -F "file=@episode.mp3"
    """
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type {file.content_type!r}. "
                f"Accepted: {sorted(_ALLOWED_MIME)}"
            ),
        )

    suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"

    # Stream upload to a temp file — avoids holding the full file in memory
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        total = 0
        while chunk := await file.read(_READ_CHUNK):
            total += len(chunk)
            if total > _MAX_BYTES:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")
            tmp.write(chunk)

    try:
        result = await run_etl(tmp_path, source_name=file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResult(**result)
