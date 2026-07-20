"""
GET /audio/{audio_id}/clip — cut a short mp3 clip from a stored source episode
via ffmpeg, cached on disk so repeat plays are instant.
"""

from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.etl.audio_store import AUDIO_DIR, enforce_budget, find_audio, touch

router = APIRouter(prefix="/audio", tags=["audio"])

CLIPS_DIR = AUDIO_DIR / "clips"
MAX_CLIP_SECONDS = 120
PAD_SECONDS = 1.0

# audio_id is always a 12-char lowercase hex sha1 prefix (see podcast_rss.py /
# upload.py) — reject anything else so a crafted id can't escape AUDIO_DIR.
_AUDIO_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _clip_path(audio_id: str, start: float, end: float) -> Path:
    return CLIPS_DIR / f"{audio_id}_{start:.2f}_{end:.2f}.mp3"


async def _extract_clip(source: Path, dest: Path, start: float, end: float) -> None:
    cut_start = max(0.0, start - PAD_SECONDS)
    duration = (end - start) + 2 * PAD_SECONDS
    dest.parent.mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-ss", f"{cut_start}",
        "-i", str(source),
        "-t", f"{duration}",
        "-c:a", "libmp3lame",
        "-q:a", "4",
        "-y",
        str(dest),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[-500:]}")


@router.get("/{audio_id}/clip")
async def get_clip(audio_id: str, start: float, end: float) -> FileResponse:
    """
    Serve a short mp3 clip cut from `[start-1s, end+1s]` of the stored source
    episode. Cached on disk keyed by (audio_id, start, end); repeat requests
    hit the cache. Returns 404 if the source episode was evicted under the
    1 GB storage cap (see audio_store.py) — the frontend should treat that
    as "clip no longer available."
    """
    if not _AUDIO_ID_RE.match(audio_id):
        raise HTTPException(status_code=404, detail="Unknown audio_id.")
    if not (math.isfinite(start) and math.isfinite(end)):
        raise HTTPException(status_code=400, detail="start/end must be finite numbers.")
    if start < 0 or end <= start:
        raise HTTPException(status_code=400, detail="Require 0 <= start < end.")
    if end - start > MAX_CLIP_SECONDS:
        raise HTTPException(
            status_code=400, detail=f"Clip length exceeds {MAX_CLIP_SECONDS}s limit."
        )

    dest = _clip_path(audio_id, start, end)

    if dest.exists():
        touch(dest)
        source = find_audio(audio_id)
        if source is not None:
            touch(source)
        return FileResponse(dest, media_type="audio/mpeg")

    source = find_audio(audio_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source audio no longer available.")

    await _extract_clip(source, dest, start, end)
    touch(source)
    enforce_budget()

    return FileResponse(dest, media_type="audio/mpeg")
