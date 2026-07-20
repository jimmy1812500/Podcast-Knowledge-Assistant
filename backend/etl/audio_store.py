"""
Persistent storage for source episode audio, keyed by audio_id.

Episodes are kept on disk after ETL (instead of deleted) so that cited
timestamps can be played back later via /audio/{audio_id}/clip. Total
storage (source episodes + cached clips) is capped at 1 GB with LRU
eviction by file mtime.
"""

from __future__ import annotations

import shutil
from pathlib import Path

AUDIO_DIR = Path("./audio_library")
MAX_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB total, source episodes + cached clips combined


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _evict_until_fits(incoming_size: int) -> None:
    """Delete least-recently-touched files (source episodes and cached clips
    share the same budget) until there's room for the incoming file."""
    while _dir_size(AUDIO_DIR) + incoming_size > MAX_BYTES:
        candidates = [f for f in AUDIO_DIR.rglob("*") if f.is_file()]
        if not candidates:
            break
        oldest = min(candidates, key=lambda f: f.stat().st_mtime)
        oldest.unlink()


def save_audio(tmp_path: Path, audio_id: str) -> Path:
    """Move a downloaded/uploaded file into permanent storage, keyed by audio_id.
    Evicts oldest-touched files first if this would exceed the 1 GB cap."""
    AUDIO_DIR.mkdir(exist_ok=True)
    dest = AUDIO_DIR / f"{audio_id}{tmp_path.suffix}"
    if dest.exists():
        tmp_path.unlink(missing_ok=True)  # already stored — dedupe by content hash
        return dest
    _evict_until_fits(tmp_path.stat().st_size)
    shutil.move(str(tmp_path), dest)
    return dest


def touch(audio_id_path: Path) -> None:
    """Mark a file as recently used so LRU eviction favors it over untouched files."""
    audio_id_path.touch(exist_ok=True)


def enforce_budget() -> None:
    """Evict oldest-touched files until total storage is back under the 1 GB cap.
    Used after writing a cached clip directly (outside save_audio's move-in path)."""
    _evict_until_fits(0)


def find_audio(audio_id: str) -> Path | None:
    """Return the stored source episode file for audio_id, or None if evicted/never stored."""
    if not AUDIO_DIR.exists():
        return None
    matches = list(AUDIO_DIR.glob(f"{audio_id}.*"))
    return matches[0] if matches else None
