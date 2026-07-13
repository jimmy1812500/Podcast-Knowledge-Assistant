"""
FastAPI router for podcast RSS ingestion.

Endpoints:
    POST /ingest/podcast            — enqueue an async ingest job via Celery
    GET  /ingest/status/{task_id}   — poll Celery task progress
"""

from __future__ import annotations

from datetime import datetime

from celery.result import AsyncResult
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from backend.etl.podcast_registry import get_podcast
from backend.worker import async_ingest_podcast

router = APIRouter(prefix="/ingest", tags=["podcast"])

# Maps task_id → user_id so the status endpoint can verify ownership.
_task_owners: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PodcastIngestRequest(BaseModel):
    podcast_id: str | None = Field(
        default=None,
        description="Registry key (e.g. 'huberman', 'lex_fridman'). Looked up in the podcast registry.",
        examples=["huberman", "lex_fridman"],
    )
    feed_url: str | None = Field(
        default=None,
        description="Custom RSS feed URL. Used as both the RSS source and the stored podcast_id when podcast_id is not provided.",
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
    dest_dir: str = Field(default="audio_cache")
    diarize: bool = Field(
        default=False,
        description="Enable speaker diarization (requires HF_TOKEN set on the worker).",
    )


class IngestAccepted(BaseModel):
    task_id: str
    status: str = "queued"


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    info: dict | str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/podcast", response_model=IngestAccepted, status_code=202)
async def ingest_podcast(
    req: PodcastIngestRequest,
    x_user_id: str | None = Header(None, alias="X-User-ID"),
) -> IngestAccepted:
    """
    Enqueue an async podcast ingest job via Celery.

    Returns a `task_id` you can poll with GET /ingest/status/{task_id}.

    Example:
        curl -X POST http://localhost:8000/ingest/podcast \\
          -H 'Content-Type: application/json' \\
          -H 'X-User-ID: alice' \\
          -d '{"podcast_id": "huberman", "max_episodes": 2, "title_filter": "sleep"}'
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header is required.")

    # Resolve (podcast_id, feed_url) from the request.
    if req.podcast_id is not None:
        try:
            entry = get_podcast(req.podcast_id)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        resolved_podcast_id = req.podcast_id
        resolved_feed_url = entry["rss"]
    elif req.feed_url is not None:
        resolved_podcast_id = req.feed_url
        resolved_feed_url = req.feed_url
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'podcast_id' (registry key) or 'feed_url' (custom RSS URL).",
        )

    since_str = req.since.isoformat() if req.since else None
    task = async_ingest_podcast.delay(
        resolved_feed_url,
        req.max_episodes,
        req.title_filter,
        since_str,
        req.dest_dir,
        x_user_id,
        req.diarize,
        resolved_podcast_id,
    )
    _task_owners[task.id] = x_user_id
    return IngestAccepted(task_id=task.id)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_status(
    task_id: str,
    x_user_id: str | None = Header(None, alias="X-User-ID"),
) -> TaskStatusResponse:
    """Poll the status of a Celery ingest task."""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header is required.")

    owner = _task_owners.get(task_id)
    if owner is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")
    if owner != x_user_id:
        raise HTTPException(status_code=403, detail="You do not own this task.")

    result = AsyncResult(task_id)
    info = result.info
    if isinstance(info, Exception):
        info = str(info)
    elif isinstance(info, dict):
        pass
    else:
        info = str(info) if info is not None else None
    return TaskStatusResponse(task_id=task_id, state=result.state, info=info)
