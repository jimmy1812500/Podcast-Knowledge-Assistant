# Phase 5 — Multi-Podcast Support

## Background

The current ingestion pipeline is fully generic, but two files hard-code Huberman Lab: `backend/fetch_huberman.py` and `backend/ingest_batch.py`. The ChromaDB metadata also has no `podcast_id` field, so there is no way to filter retrieval by show.

This phase introduces a podcast registry and threads `podcast_id` through the entire stack — from the ingest request, through the ETL metadata, to the retrieval `where` filter — turning the assistant into a general multi-podcast tool.

**Scope:** new `backend/etl/podcast_registry.py`; modifications to `backend/etl/embeddings.py`, `backend/routers/podcast.py`, `backend/agents/state.py`, `backend/agents/retrieval.py`, `backend/routers/chat.py`.

---

## Tasks

### 1. Create `backend/etl/podcast_registry.py`
- [ ] Define a `PODCASTS: dict[str, dict]` constant with at least two entries:
  - `"huberman"` → `{"name": "Huberman Lab", "rss": "https://feeds.megaphone.fm/hubermanlab"}`
  - `"lex_fridman"` → `{"name": "Lex Fridman Podcast", "rss": "https://lexfridman.com/feed/podcast/"}`
- [ ] Add a helper `get_podcast(podcast_id: str) -> dict` that raises `KeyError` with a clear message if the ID is unknown

### 2. Modify `backend/etl/embeddings.py` — `store_chunks()`
- [ ] Accept `podcast_id: str = "unknown"` as a parameter
- [ ] Add `"podcast_id": podcast_id` to the ChromaDB metadata dict stored for every chunk
- [ ] Update all call sites of `store_chunks()` in the ETL pipeline to pass the podcast ID

### 3. Modify `backend/routers/podcast.py`
- [ ] Add `podcast_id: str` to the ingest request body (replace any hardcoded Huberman reference)
- [ ] Validate `podcast_id` against `podcast_registry.PODCASTS`; return HTTP 422 if unknown
- [ ] Look up the RSS URL via `podcast_registry.get_podcast(podcast_id)["rss"]` and pass it to the pipeline

### 4. Modify `backend/agents/state.py`
- [ ] Add `podcast_filter: str | None` to `AgentState`

### 5. Modify `backend/agents/retrieval.py`
- [ ] Read `state.get("podcast_filter")` inside `_query_chroma_sync`
- [ ] When it is not `None`, add `where={"podcast_id": podcast_filter}` to `collection.query()`
- [ ] When it is `None`, omit the `where` clause so the search spans all podcasts

### 6. Modify `backend/routers/chat.py`
- [ ] Add `podcast_filter: str | None = None` to `ChatRequest`
- [ ] Thread `podcast_filter` into `init_state["podcast_filter"]`

---

## Acceptance Criteria

- [ ] `backend/etl/podcast_registry.py` exists with at least two podcast entries
- [ ] `POST /ingest/podcast` with `{"podcast_id": "lex_fridman"}` triggers ingestion of the Lex Fridman RSS feed without errors
- [ ] `POST /ingest/podcast` with an unknown `podcast_id` returns HTTP 422
- [ ] Chunks ingested for different podcasts have different `podcast_id` values in ChromaDB metadata (verified via direct `collection.get()`)
- [ ] `POST /chat` with `{"query": "...", "podcast_filter": "huberman"}` only returns sources where `source` belongs to Huberman Lab episodes
- [ ] `POST /chat` with no `podcast_filter` returns sources from all ingested podcasts
- [ ] `backend/fetch_huberman.py` and `backend/ingest_batch.py` either deleted or refactored to use the registry (no hardcoded RSS URLs remain outside the registry)

---

## Notes

- If Phase 4.5 is already merged, pass `podcast_id` as an argument to the Celery task and include it in the `user_id`-scoped metadata write.
- ChromaDB `where` filters use exact string match — `podcast_id` values must be consistent between ingestion and query time (use the registry key, e.g. `"huberman"`, not the display name).
- Adding a new podcast is a one-liner in `podcast_registry.py` — no code changes elsewhere.
