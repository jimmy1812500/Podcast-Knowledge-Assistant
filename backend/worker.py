"""
Celery worker for asynchronous podcast ingestion.

Start locally (with redis-server running):
    uv run celery -A backend.worker.celery_app worker --loglevel=info --concurrency=1
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("podcast_worker", broker=REDIS_URL, backend=REDIS_URL)


async def _run_pipeline_async(
    feed_url: str,
    max_episodes: int,
    title_filter: str | None,
    since: datetime | None,
    dest_dir: str,
    user_id: str,
    diarize: bool,
    podcast_id: str = "unknown",
) -> dict:
    from backend.etl.pipeline import run_etl
    from backend.etl.podcast_rss import download_episodes, fetch_episodes

    dest = Path(dest_dir)
    episodes = await fetch_episodes(
        feed_url,
        max_episodes=max_episodes,
        title_filter=title_filter,
        since=since,
    )
    paths = await download_episodes(episodes, dest_dir=dest)

    results = []
    for ep, path in zip(episodes, paths):
        try:
            etl_result = await run_etl(
                path, source_name=ep.title, user_id=user_id, diarize=diarize, podcast_id=podcast_id
            )
            results.append({"title": ep.title, "status": "done", "etl": etl_result})
        except Exception as exc:  # noqa: BLE001
            results.append({"title": ep.title, "status": "error", "error": str(exc)})
        finally:
            path.unlink(missing_ok=True)

    return {"episodes_processed": len(results), "results": results}


@celery_app.task(bind=True, name="worker.async_ingest_podcast")
def async_ingest_podcast(
    self,
    feed_url: str,
    max_episodes: int = 3,
    title_filter: str | None = None,
    since: str | None = None,
    dest_dir: str = "/tmp/podcasts",
    user_id: str = "public",
    diarize: bool = False,
    podcast_id: str = "unknown",
) -> dict:
    self.update_state(state="PROGRESS", meta={"status": "starting"})
    since_dt = datetime.fromisoformat(since) if since else None
    return asyncio.run(
        _run_pipeline_async(
            feed_url=feed_url,
            max_episodes=max_episodes,
            title_filter=title_filter,
            since=since_dt,
            dest_dir=dest_dir,
            user_id=user_id,
            diarize=diarize,
            podcast_id=podcast_id,
        )
    )
