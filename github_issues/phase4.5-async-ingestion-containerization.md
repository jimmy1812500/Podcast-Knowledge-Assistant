# Phase 4.5 — Production-Grade Asynchronous Ingestion & Containerization

## Background

The current ingestion process (ASR via `faster-whisper` + optional LLM extraction) is CPU-intensive and can take 3–4 minutes per episode. Running it synchronously inside a FastAPI route will cause HTTP timeouts and block the server event loop for all other requests.

Additionally, the project currently has no environment isolation — dependencies, the ChromaDB data directory, and audio files are all mixed into the local development environment, making deployment fragile.

This phase introduces three things:
1. **Celery + Redis** to offload ingestion to a background worker process
2. **Docker Compose** to containerise all services for reproducible deployment
3. **Multi-tenant data isolation** via `user_id` scoping in ChromaDB metadata

**Scope:** new `Dockerfile`, `docker-compose.yml`, `backend/worker.py`; modifications to `backend/routers/podcast.py`, `backend/etl/embeddings.py`, `backend/agents/retrieval.py`, `backend/agents/state.py`, `backend/routers/chat.py`.

---

## Tasks

### 1. Add new dependencies
- [ ] Add `celery>=5.4.0` and `redis>=5.2.0` to `pyproject.toml` under `[project] dependencies`
- [ ] Run `uv sync` to lock them

### 2. Create `Dockerfile` at project root
- [ ] Base image: `python:3.11-slim`
- [ ] `WORKDIR /app`
- [ ] Install `uv` via `pip install uv`
- [ ] Copy `pyproject.toml` and `uv.lock`, then run `uv sync --frozen --no-dev`
- [ ] Copy `backend/` directory into the image

### 3. Create `docker-compose.yml` at project root
- [ ] Define four services: `web`, `celery_worker`, `redis`, `chromadb`
- [ ] `web`: builds from `Dockerfile`, runs `uvicorn backend.main:app --host 0.0.0.0 --port 8000`, exposes port `8000`, depends on `redis` and `chromadb`
- [ ] `celery_worker`: same build, runs `celery -A backend.worker.celery_app worker --loglevel=info --concurrency=1`, depends on `redis` and `chromadb`
- [ ] `redis`: uses `redis:7-alpine`, exposes port `6379`
- [ ] `chromadb`: uses `chromadb/chroma:latest`, exposes port `8001`
- [ ] Mount `./backend/chroma_data` and `./audio_cache` as volumes on both `web` and `celery_worker`

### 4. Create `backend/worker.py`
- [ ] Instantiate a `Celery` app named `"podcast_worker"` using `REDIS_URL` env var as both broker and backend (default `redis://localhost:6379/0`)
- [ ] Implement `async_ingest_podcast(self, podcast_id, since=None)` as a bound Celery task:
  - Call `self.update_state(state="PROGRESS", meta={"status": "starting"})` at the start
  - Run the ETL pipeline via `asyncio.run(run_pipeline(...))`
  - Return a summary dict on success

### 5. Modify `backend/routers/podcast.py`
- [ ] Replace the existing synchronous/`BackgroundTasks` ingestion call with `async_ingest_podcast.delay(podcast_id, since)`
- [ ] Change the endpoint to return HTTP 202 with `{"task_id": task.id, "status": "queued"}`
- [ ] Add a new `GET /ingest/status/{task_id}` endpoint that queries `AsyncResult(task_id)` and returns `state` + `info`

### 6. Add `REDIS_URL` to `.env.example`
- [ ] Add the line: `REDIS_URL=redis://localhost:6379/0`

### 7. Implement multi-tenant data isolation

#### 7a. `backend/etl/embeddings.py` — `store_chunks()`
- [ ] Accept `user_id: str = "public"` as a parameter
- [ ] Store `"user_id": user_id` in the ChromaDB metadata dict for every chunk

#### 7b. `backend/agents/state.py`
- [ ] Add `user_id: str` to `AgentState`

#### 7c. `backend/agents/retrieval.py`
- [ ] Read `state["user_id"]` inside `_query_chroma_sync`
- [ ] Add `where={"user_id": state["user_id"]}` to the `collection.query()` call so each user only retrieves their own chunks

#### 7d. `backend/routers/chat.py`
- [ ] Extract `user_id` from an `X-User-ID` request header (or Bearer token)
- [ ] Inject it into `init_state["user_id"]`
- [ ] Return HTTP 401 if `user_id` is missing or empty

#### 7e. `backend/routers/podcast.py`
- [ ] Pass `user_id` to the Celery task arguments so every ingested chunk is tagged with the correct owner

---

## Acceptance Criteria

- [ ] `celery` and `redis` appear in `uv.lock` after `uv sync`
- [ ] `Dockerfile` builds successfully: `docker build -t podcast-assistant .`
- [ ] `docker-compose up` starts all four services (`web`, `celery_worker`, `redis`, `chromadb`) without errors
- [ ] `POST /ingest/podcast` returns HTTP 202 and a `task_id` within 1 second (no blocking)
- [ ] `GET /ingest/status/{task_id}` returns `state: "PENDING"` immediately after dispatch and `state: "SUCCESS"` after the worker completes
- [ ] Chunks stored during ingestion have a `user_id` field in their ChromaDB metadata (verify with a direct ChromaDB `get()` call)
- [ ] `POST /chat` without `X-User-ID` header returns HTTP 401
- [ ] A user can only retrieve chunks tagged with their own `user_id` — querying with a different `user_id` returns no results for the same question
- [ ] `GET /health` on the `web` service returns `{"status": "ok"}` when running via Docker Compose

---

## Notes

- During local development (no Docker), you can test Celery by running `redis-server` separately and `uv run celery -A backend.worker.celery_app worker --loglevel=info` in a second terminal.
- Set `--concurrency=1` on the worker to avoid parallel CPU contention between Whisper and embedding model loads.
- Add `./audio_cache/` and `./backend/chroma_data/` to `.gitignore` if not already present.
