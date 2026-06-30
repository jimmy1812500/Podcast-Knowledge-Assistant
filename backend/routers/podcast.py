"""
FastAPI router for podcast RSS ingestion.

Endpoints:
    POST /ingest/podcast        — kick off a background ingest job
    GET  /ingest/status/{id}    — poll job progress
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from backend.etl.pipeline import run_etl
from backend.etl.podcast_rss import HUBERMAN_LAB_RSS, Episode, download_episodes, fetch_episodes

router = APIRouter(prefix="/ingest", tags=["podcast"])

# ---------------------------------------------------------------------------
# In-memory job store (replace with PostgreSQL in production)
# ---------------------------------------------------------------------------

JobStatus = Literal["queued", "fetching", "downloading", "processing", "done", "error"]

_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PodcastIngestRequest(BaseModel):
    feed_url: str = Field(
        default=HUBERMAN_LAB_RSS,
        description="RSS feed URL. Defaults to Huberman Lab.",
    )
    max_episodes: int = Field(default=3, ge=1, le=20)
    title_filter: str | None = Field(
        default=None,
        description="Case-insensitive substring to match episode titles.",
        examples=["dopamine", "sleep", "exercise"],
    )
    since: datetime | None = Field(
        default=None,
        description="ISO-8601 datetime — only ingest episodes published after this.",
    )
    dest_dir: str = Field(default="/tmp/podcasts")


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    episodes: list[dict] = []
    error: str | None = None


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------


async def _run_pipeline(job_id: str, req: PodcastIngestRequest) -> None:
    job = _jobs[job_id]
    dest = Path(req.dest_dir)

    try:
        job["status"] = "fetching"
        episodes: list[Episode] = await fetch_episodes(
            req.feed_url,
            max_episodes=req.max_episodes,
            title_filter=req.title_filter,
            since=req.since,
        )
        job["episodes"] = [
            {
                "episode_id": ep.episode_id,
                "title": ep.title,
                "published": ep.published.isoformat(),
                "duration": ep.duration,
                "audio_url": ep.audio_url,
                "status": "queued",
            }
            for ep in episodes
        ]

        job["status"] = "downloading"
        paths = await download_episodes(episodes, dest_dir=dest)

        for i, path in enumerate(paths):
            job["episodes"][i]["local_path"] = str(path)
            job["episodes"][i]["status"] = "downloaded"

        job["status"] = "processing"
        for i, (ep, path) in enumerate(zip(episodes, paths)):
            try:
                etl_result = await run_etl(path, source_name=ep.title)
                job["episodes"][i]["etl"] = etl_result
                job["episodes"][i]["status"] = "done"
            except Exception as ep_exc:  # noqa: BLE001
                job["episodes"][i]["status"] = "error"
                job["episodes"][i]["error"] = str(ep_exc)
            finally:
                path.unlink(missing_ok=True)  # free disk space after ETL

        job["status"] = "done"

    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/podcast", response_model=JobResponse, status_code=202)
async def ingest_podcast(
    req: PodcastIngestRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """
    Kick off an async podcast ingest job.

    Returns a `job_id` you can poll with GET /ingest/status/{job_id}.

    Example — ingest 2 Huberman Lab episodes about sleep:

        curl -X POST http://localhost:8000/ingest/podcast \\
          -H 'Content-Type: application/json' \\
          -d '{"max_episodes": 2, "title_filter": "sleep"}'
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "episodes": [], "error": None}
    background_tasks.add_task(_run_pipeline, job_id, req)
    return JobResponse(job_id=job_id, status="queued")


@router.get("/status/{job_id}", response_model=JobResponse)
async def get_status(job_id: str) -> JobResponse:
    """Poll the status of a podcast ingest job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return JobResponse(
        job_id=job_id,
        status=job["status"],
        episodes=job["episodes"],
        error=job.get("error"),
    )
