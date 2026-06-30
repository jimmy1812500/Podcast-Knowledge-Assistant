"""
Podcast RSS Aggregator — fetches, filters, and downloads podcast episodes.

Usage:
    episodes = await fetch_episodes(HUBERMAN_LAB_RSS)
    paths = await download_episodes(episodes[:3], dest_dir=Path("/tmp/podcasts"))
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import httpx

HUBERMAN_LAB_RSS = "https://feeds.megaphone.fm/hubermanlab"

_CHUNK_SIZE = 64 * 1024  # 64 KB per read


@dataclass
class Episode:
    title: str
    audio_url: str
    published: datetime
    duration: str
    description: str
    episode_id: str  # stable hash of the audio URL


def _parse_date(entry) -> datetime:
    if hasattr(entry, "published"):
        try:
            dt = parsedate_to_datetime(entry.published)
            # -0000 / +0000 → naive datetime; treat as UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _find_audio_url(entry) -> str | None:
    for link in getattr(entry, "enclosures", []):
        if "audio" in link.get("type", ""):
            return link.get("href") or link.get("url")
    for link in getattr(entry, "links", []):
        if "audio" in link.get("type", "") or link.get("href", "").endswith(".mp3"):
            return link.get("href")
    return None


def _episode_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


def parse_feed(feed_url: str) -> list[Episode]:
    """Synchronously parse an RSS feed and return episodes, newest first."""
    parsed = feedparser.parse(feed_url)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"Failed to parse feed: {feed_url!r}")

    episodes: list[Episode] = []
    for entry in parsed.entries:
        url = _find_audio_url(entry)
        if not url:
            continue
        episodes.append(
            Episode(
                title=entry.get("title", "Untitled"),
                audio_url=url,
                published=_parse_date(entry),
                duration=entry.get("itunes_duration", ""),
                description=re.sub(r"<[^>]+>", "", entry.get("summary", "")),
                episode_id=_episode_id(url),
            )
        )

    episodes.sort(key=lambda e: e.published, reverse=True)
    return episodes


async def fetch_episodes(
    feed_url: str = HUBERMAN_LAB_RSS,
    *,
    max_episodes: int | None = None,
    title_filter: str | None = None,
    since: datetime | None = None,
) -> list[Episode]:
    """
    Async wrapper around parse_feed with optional filtering.

    Args:
        feed_url: RSS feed URL.
        max_episodes: Cap on number of episodes returned.
        title_filter: Case-insensitive substring match on episode title.
        since: Only return episodes published after this datetime.
    """
    loop = asyncio.get_event_loop()
    episodes = await loop.run_in_executor(None, parse_feed, feed_url)

    if since:
        episodes = [e for e in episodes if e.published > since]
    if title_filter:
        kw = title_filter.lower()
        episodes = [e for e in episodes if kw in e.title.lower()]
    if max_episodes:
        episodes = episodes[:max_episodes]

    return episodes


async def _download_one(
    client: httpx.AsyncClient,
    episode: Episode,
    dest_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    safe_name = re.sub(r"[^\w\-.]", "_", episode.title)[:80]
    dest = dest_dir / f"{episode.episode_id}_{safe_name}.mp3"

    if dest.exists() and not overwrite:
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)

    async with client.stream("GET", episode.audio_url, follow_redirects=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            async for chunk in resp.aiter_bytes(_CHUNK_SIZE):
                fh.write(chunk)

    return dest


async def download_episodes(
    episodes: list[Episode],
    dest_dir: Path = Path("/tmp/podcasts"),
    *,
    concurrency: int = 2,
    overwrite: bool = False,
    timeout_seconds: int = 600,
) -> list[Path]:
    """
    Download a list of episodes concurrently, returning local file paths.

    Args:
        episodes: Episodes to download (from fetch_episodes).
        dest_dir: Directory where MP3 files are saved.
        concurrency: Max simultaneous downloads.
        overwrite: Re-download even if the file already exists.
        timeout_seconds: Per-request timeout.
    """
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(timeout_seconds)

    async def _bounded(ep: Episode) -> Path:
        async with semaphore:
            return await _download_one(client, ep, dest_dir, overwrite=overwrite)

    async with httpx.AsyncClient(timeout=timeout) as client:
        paths = await asyncio.gather(*[_bounded(ep) for ep in episodes])

    return list(paths)
