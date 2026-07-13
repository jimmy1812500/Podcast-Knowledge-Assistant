"""
GET /audio/{filename} — serve retained MP3 files from the audio_cache directory.

Path traversal is blocked: any filename that resolves outside AUDIO_DIR returns HTTP 400.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# Absolute path anchored to this file's location so the server start directory
# doesn't matter.
AUDIO_DIR = (Path(__file__).parent.parent / "audio_cache").resolve()

router = APIRouter(prefix="/audio", tags=["audio"])


@router.get("/{filename}")
async def get_audio(filename: str) -> FileResponse:
    """Stream an MP3 file from the audio cache by filename.

    Returns HTTP 400 for path traversal attempts, HTTP 404 if not found.
    """
    target = (AUDIO_DIR / filename).resolve()

    if not target.is_relative_to(AUDIO_DIR):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"{filename!r} not found in audio cache.")

    return FileResponse(target, media_type="audio/mpeg")
