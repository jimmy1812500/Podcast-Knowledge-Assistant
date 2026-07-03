# Deepening the Podcast Knowledge Assistant

This document is a concrete implementation roadmap for extending beyond Phase 1 (ETL) and Phase 2 (RAG agent). Each phase is self-contained and ordered by impact-to-effort ratio.

---

## Current Architecture Snapshot

```
ETL:   RSS → faster-whisper → splitter (500-char) → BGE embed → ChromaDB
Agent: retrieve (top-5 cosine) → evaluate → expand → synthesize (gemma3:4b)
API:   POST /chat  |  GET /chat/stream (SSE)  |  POST /upload  |  POST /ingest/podcast
```

---

## Phase 3 — Cross-Encoder Re-ranking

**Why:** Top-5 cosine retrieval ranks by embedding similarity, which captures semantics but misses precision. A cross-encoder reads the query and each candidate together and re-scores them with far higher accuracy. This is the single highest-leverage retrieval improvement.

**New dependency:**
```
sentence-transformers  # already installed — cross-encoder support is built in
```

**New file:** `backend/agents/reranker.py`

```python
from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None

def _load() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model

def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Score each chunk against the query and return top_k by cross-encoder score."""
    model = _load()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]
```

**Modify:** `backend/agents/retrieval.py`
- Increase `TOP_K = 10` (retrieve more candidates for the re-ranker to filter)
- After `_query_chroma_sync`, call `rerank(query, context, top_k=3)` before returning

**Modify:** `backend/agents/graph.py` — no changes needed; the reranker is transparent to the graph

**Trade-off:** `ms-marco-MiniLM-L-6-v2` is ~85 MB and runs in ~50 ms per re-rank pass on CPU. The heavier `ms-marco-MiniLM-L-12-v2` is more accurate but ~2× slower. Start with the L-6 variant.

---

## Phase 4 — Multi-turn Conversation Memory

**Why:** Every `/chat` call today starts from a blank `AgentState`. Users cannot ask follow-up questions ("what else did he say about that?") because the agent has no memory of the prior exchange. LangGraph supports message history natively via `add_messages`.

**Modify:** `backend/agents/state.py`

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    query: str
    search_query: str
    context: list[dict]
    evaluation: str
    answer: str
    iteration: int
    messages: Annotated[list, add_messages]  # add this field
```

**Modify:** `backend/agents/synthesis.py`
- Pass `state["messages"]` as the conversation history when calling Ollama, so the LLM can see prior Q&A pairs before generating the answer.

**Modify:** `backend/routers/chat.py`
- Add `session_id: str | None` to `ChatRequest`
- Keep an in-memory `dict[str_session_id, list[messages]]` store (or use `langgraph.checkpoint.memory.MemorySaver` for the graph)
- On each request, load prior messages by session_id, attach to `init_state`, and persist the updated messages back after the graph run

**Modify:** `backend/static/index.html`
- Generate a random `session_id` on page load (e.g. `crypto.randomUUID()`) and include it in every request body

**New dependency:**
```
# No new package needed — MemorySaver ships with langgraph
```

**Checkpoint approach (cleaner alternative):**
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = build_graph().compile(checkpointer=checkpointer)

# In chat endpoint:
config = {"configurable": {"thread_id": req.session_id or "default"}}
final = await graph.ainvoke(init_state, config=config)
```
This delegates all history bookkeeping to LangGraph — no manual message list management needed.

---

## Phase 4.5 — Production-Grade Asynchronous Ingestion & Containerization

**Why:** The current ingestion process (ASR + LLM extraction) is highly blocking and resource-intensive. Running it synchronously inside FastAPI routes will cause client timeouts for any non-trivial episode. Furthermore, the lack of environment isolation makes deployment risky. Introducing Celery, Redis, and Docker resolves both problems: long-running jobs run in a separate worker process, the API stays responsive, and every service runs in a reproducible container.

**New dependencies:**
```
celery>=5.4.0
redis>=5.2.0
```

**New file:** `docker-compose.yml` (project root)

```yaml
services:
  web:
    build: .
    command: uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file: backend/.env
    depends_on:
      - redis
      - chromadb
    volumes:
      - ./backend/chroma_data:/app/backend/chroma_data
      - ./audio_cache:/app/audio_cache

  celery_worker:
    build: .
    command: uv run celery -A backend.worker.celery_app worker --loglevel=info --concurrency=1
    env_file: backend/.env
    depends_on:
      - redis
      - chromadb
    volumes:
      - ./backend/chroma_data:/app/backend/chroma_data
      - ./audio_cache:/app/audio_cache

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8001"
    volumes:
      - ./backend/chroma_data:/chroma/chroma
```

**New file:** `Dockerfile` (project root)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY backend/ ./backend/
```

**New file:** `backend/worker.py`

```python
from celery import Celery
import os

celery_app = Celery(
    "podcast_worker",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

@celery_app.task(bind=True, name="ingest_podcast")
def async_ingest_podcast(self, podcast_id: str, since: str | None = None) -> dict:
    """Run the full ETL pipeline in a Celery worker process."""
    import asyncio
    from backend.etl.pipeline import run_pipeline

    self.update_state(state="PROGRESS", meta={"status": "starting", "episode": None})
    result = asyncio.run(run_pipeline(podcast_id=podcast_id, since=since, progress_hook=self.update_state))
    return result
```

**Modify:** `backend/routers/podcast.py`
- Replace the synchronous/background-task ingestion call with a Celery `.delay()` dispatch:
  ```python
  from backend.worker import async_ingest_podcast

  @router.post("/ingest/podcast", status_code=202)
  async def ingest_podcast(req: IngestRequest):
      task = async_ingest_podcast.delay(req.podcast_id, req.since)
      return {"task_id": task.id, "status": "queued"}
  ```
- The endpoint returns HTTP 202 immediately — no timeout risk regardless of episode length.

**New endpoint:** `GET /ingest/status/{task_id}` (add to `backend/routers/podcast.py`)
```python
from celery.result import AsyncResult
from backend.worker import celery_app

@router.get("/ingest/status/{task_id}")
async def ingest_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "state": result.state,          # PENDING | PROGRESS | SUCCESS | FAILURE
        "info": result.info or {},
    }
```

**Action 3 — Multi-tenant data isolation:**
Add a `user_id` field to the ChromaDB metadata stored per chunk and enforce it as a filter at query time.

- **Modify:** `backend/etl/embeddings.py` — `store_chunks()`
  ```python
  "user_id": c.get("user_id", "public"),
  ```
- **Modify:** `backend/agents/retrieval.py` — add `user_id` to `AgentState` and pass it as a mandatory `where` filter:
  ```python
  where={"user_id": state["user_id"]}
  ```
- **Modify:** `backend/agents/state.py` — add `user_id: str`
- **Modify:** `backend/routers/chat.py` — extract `user_id` from the request (e.g. a Bearer token or `X-User-ID` header) and inject it into `init_state`. Reject requests with no `user_id`.
- **Modify:** `backend/routers/podcast.py` — bind the `user_id` to the Celery task arguments so every chunk ingested by that job is tagged with the correct owner.

**Add to `.env.example`:**
```
REDIS_URL=redis://localhost:6379/0
```

**Trade-off:** Celery + Redis adds operational complexity (two extra processes to run locally). During development, you can still run ingestion synchronously by calling `async_ingest_podcast(podcast_id, since)` directly (no `.delay()`). Use `docker-compose up` only when testing the full production stack.

---

## Phase 5 — Multi-Podcast Support

**Why:** The ingestion pipeline is already generic; only `backend/fetch_huberman.py` and `backend/ingest_batch.py` are Huberman-specific. Parameterizing the feed URL turns this into a general podcast assistant.

**New file:** `backend/etl/podcast_registry.py`

```python
PODCASTS: dict[str, dict] = {
    "huberman": {
        "name": "Huberman Lab",
        "rss": "https://feeds.megaphone.fm/hubermanlab",
    },
    "lex_fridman": {
        "name": "Lex Fridman Podcast",
        "rss": "https://lexfridman.com/feed/podcast/",
    },
    # add more entries here
}
```

**Modify:** `backend/etl/embeddings.py`
- Add `podcast_id` to the ChromaDB metadata stored per chunk:
  ```python
  "podcast_id": c.get("podcast_id", "unknown"),
  ```

**Modify:** `backend/agents/retrieval.py`
- Accept an optional `podcast_filter: str | None` in the state
- Pass it as a `where={"podcast_id": podcast_filter}` filter to `collection.query()` when set

**Modify:** `backend/routers/chat.py`
- Add `podcast_filter: str | None = None` to `ChatRequest`
- Thread it through to the initial `AgentState`

**Modify:** `backend/routers/podcast.py`
- Accept `podcast_id` in the ingest request body instead of hardcoding Huberman
- Look up the RSS URL from `podcast_registry.PODCASTS[podcast_id]`

**Modify:** `AgentState` in `backend/agents/state.py`
- Add `podcast_filter: str | None`

---

## Phase 6 — Audio Timestamp Playback in the UI

**Why:** Every retrieved chunk already carries `timestamp_start` and `timestamp_end` (seconds). Wiring those to an HTML5 `<audio>` player lets users jump directly to the spoken moment — this closes the loop from text answer back to the original audio.

**Modify:** `backend/static/index.html`
- Add a hidden `<audio id="player">` element
- When rendering a source card, add a "▶ Play" button that:
  ```javascript
  const audio = document.getElementById('player');
  audio.src = `/audio/${encodeURIComponent(source.source)}`;
  audio.currentTime = source.timestamp_start;
  audio.play();
  ```

**New endpoint:** `backend/routers/upload.py` (or a new `audio.py` router)
```python
@router.get("/audio/{filename}")
async def serve_audio(filename: str) -> FileResponse:
    path = AUDIO_DIR / filename  # AUDIO_DIR = where MP3s are downloaded
    return FileResponse(path, media_type="audio/mpeg")
```

**Prerequisite:** The MP3 files must be retained after ingestion. Check `backend/etl/podcast_rss.py` — if it deletes files post-transcription, change that to keep them (or re-download on demand).

**Storage note:** MP3s are large (~100 MB/episode). Consider storing them in a dedicated `./audio_cache/` directory and adding it to `.gitignore`.

---

## Phase 7 — Hybrid Search (BM25 + Vector)

**Why:** Cosine search retrieves semantically similar text but can miss exact keyword matches (drug names, study authors, specific numbers). BM25 catches these. Combining both (reciprocal rank fusion) produces better recall than either alone.

**New dependency:**
```
rank-bm25>=0.2.2
```

**New file:** `backend/agents/bm25_index.py`

```python
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, q: str, top_k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(q.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self.chunks[i] for i, _ in ranked[:top_k]]
```

**Strategy — Reciprocal Rank Fusion (RRF):**
```python
def rrf_merge(vector_hits: list[dict], bm25_hits: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    id_to_chunk: dict[str, dict] = {}
    for rank, chunk in enumerate(vector_hits):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        id_to_chunk[cid] = chunk
    for rank, chunk in enumerate(bm25_hits):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        id_to_chunk[cid] = chunk
    return [id_to_chunk[cid] for cid in sorted(scores, key=scores.__getitem__, reverse=True)]
```

**Modify:** `backend/agents/retrieval.py`
- Run vector search and BM25 search in parallel
- Merge with `rrf_merge`
- Optionally apply re-ranker (Phase 3) on top of merged results

**Build the BM25 index at startup:** Load all chunk texts from ChromaDB once, build the `BM25Index`, and keep it as a singleton (similar to `_chroma` and `_model` in `embeddings.py`). Rebuild only when new episodes are ingested.

---

## Phase 8 — Semantic Chunking

**Why:** The current fixed-size 500-char splitter in `backend/etl/splitter.py` sometimes cuts mid-thought. Semantic chunking groups sentences where the embedding similarity between adjacent sentences is high, and splits where it drops — producing cleaner, topic-coherent chunks that retrieve better.

**New dependency:**
```
nltk>=3.9          # sentence tokenization
numpy>=1.26        # cosine similarity between adjacent sentence embeddings
```

**New file:** `backend/etl/semantic_splitter.py`

```python
import numpy as np
import nltk
from backend.etl.embeddings import _encode_sync  # sync encoder

nltk.download("punkt", quiet=True)

def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def semantic_split(
    text: str,
    breakpoint_threshold: float = 0.3,  # drop in cosine → new chunk
    min_chunk_chars: int = 200,
) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 1:
        return [text]

    embeddings = _encode_sync(sentences)
    similarities = [_cosine(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

    chunks, current = [], [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < breakpoint_threshold and len(" ".join(current)) >= min_chunk_chars:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))
    return chunks
```

**Modify:** `backend/etl/pipeline.py`
- Add a flag `use_semantic_chunking: bool = False`
- When `True`, call `semantic_split` instead of `split_text` from `splitter.py`

**Trade-off:** Semantic chunking requires one extra embedding pass over all sentences during ingestion (slow on CPU for long episodes). Keep the fixed splitter as the default; use semantic splitting for re-ingestion or new high-quality episodes.

---

## Phase 9 — Topic & Entity Extraction (Knowledge Layer)

**Why:** Storing raw transcript chunks is "level 0". Extracting topics, claims, and named entities during ingestion adds a structured layer that enables queries like "list all episodes that mention cortisol" or "what claims were made about protein timing?" — queries that pure semantic search handles poorly.

**Approach:** Run a local Ollama pass per chunk during ingestion to extract structured metadata.

**New file:** `backend/etl/extractor.py`

```python
import ollama, json

_PROMPT = """
Extract from this podcast transcript excerpt:
1. topics: list of 1-3 word topic tags (e.g. "sleep", "cortisol", "protein intake")
2. entities: named people, supplements, studies, or tools mentioned
3. claims: short factual claims made (one sentence each, max 3)

Return ONLY valid JSON: {{"topics": [...], "entities": [...], "claims": [...]}}

Transcript:
{text}
"""

async def extract_metadata(text: str) -> dict:
    client = ollama.AsyncClient()
    resp = await client.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
        format="json",
    )
    try:
        return json.loads(resp.message.content)
    except json.JSONDecodeError:
        return {"topics": [], "entities": [], "claims": []}
```

**Modify:** `backend/etl/embeddings.py` — `store_chunks()`
- Accept optional `topics`, `entities` lists per chunk
- Store them as ChromaDB metadata (ChromaDB supports string metadata; join lists with `,`)

**New endpoint:** `GET /topics` — aggregate all `topics` metadata values from ChromaDB to return a tag cloud / episode-topic matrix.

**Query-time use:** In `backend/agents/retrieval.py`, when the user query contains a known topic tag, add a `where={"topics": {"$contains": tag}}` pre-filter before vector search to narrow the search space.

---

## Recommended Implementation Order

| Phase | Feature | Effort | Impact |
|-------|---------|--------|--------|
| 3 | Cross-encoder re-ranking | Low (1 file, ~40 lines) | High — better answers immediately |
| 4 | Multi-turn conversation | Medium (state + checkpoint) | High — transforms UX |
| 4.5 | Async ingestion + Docker | Medium-High (Celery + Redis + Compose) | High — production-ready, no timeouts |
| 6 | Audio timestamp playback | Medium (1 endpoint + UI) | High — compelling demo |
| 5 | Multi-podcast support | Medium (registry + metadata) | Medium |
| 7 | Hybrid BM25 + vector | Medium-High | Medium |
| 8 | Semantic chunking | Medium | Medium — best for re-ingestion |
| 9 | Topic extraction | High (slow LLM pass at ingest) | Medium-High |

Start with **Phase 3** (re-ranking) — it is additive, fully local, and touches only `retrieval.py`. Then **Phase 4** (conversation memory) via LangGraph's `MemorySaver` for maximum UX gain with minimal risk. Tackle **Phase 4.5** before any public deployment to eliminate timeout risk and enforce data isolation.
