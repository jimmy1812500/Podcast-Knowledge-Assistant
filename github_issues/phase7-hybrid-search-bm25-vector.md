# Phase 7 — Hybrid Search (BM25 + Vector)

## Background

The current retrieval in `backend/agents/retrieval.py` is pure vector search (BGE cosine similarity). Cosine search excels at semantic meaning but can miss exact keyword matches — supplement names, study citations, specific numbers, or guest names that may not appear in the nearest-neighbour embedding neighbourhood.

BM25 is a classic term-frequency keyword search that catches exact matches vector search misses. Combining both via **Reciprocal Rank Fusion (RRF)** produces better recall than either alone, without needing to choose between them.

**Scope:** new `backend/agents/bm25_index.py`; modifications to `backend/agents/retrieval.py`.

---

## Tasks

### 1. Add new dependency
- [ ] Add `rank-bm25>=0.2.2` to `pyproject.toml` under `[project] dependencies`
- [ ] Run `uv sync` to lock it

### 2. Create `backend/agents/bm25_index.py`

#### 2a. `BM25Index` class
- [ ] Constructor takes `chunks: list[dict]` (same schema as ChromaDB query results)
- [ ] Tokenise each chunk text as `.lower().split()` and pass to `BM25Okapi`
- [ ] Implement `query(self, q: str, top_k: int = 10) -> list[dict]`:
  - Score all chunks with `self.bm25.get_scores(q.lower().split())`
  - Return the `top_k` chunks sorted by score descending

#### 2b. Singleton loader
- [ ] Implement `get_bm25_index() -> BM25Index` that:
  - On first call, loads **all** chunks from the ChromaDB collection (using `collection.get()` with `include=["documents", "metadatas"]`)
  - Builds and caches a `BM25Index` instance
  - Exposes a `rebuild()` function to invalidate and rebuild the index (call after new ingestion)

#### 2c. RRF merge function
- [ ] Implement `rrf_merge(vector_hits: list[dict], bm25_hits: list[dict], k: int = 60) -> list[dict]`:
  - Assign RRF scores: for each list, score for rank `r` = `1 / (k + r + 1)`
  - Accumulate scores by `chunk_id` across both lists
  - Return chunks sorted by total RRF score descending

### 3. Modify `backend/agents/retrieval.py`
- [ ] Import `get_bm25_index` and `rrf_merge` from `backend.agents.bm25_index`
- [ ] In `retrieve_node`, run vector search (existing) and BM25 search in parallel using `asyncio.gather`:
  - Vector: existing `_query_chroma_sync` call (keep `TOP_K = 10`)
  - BM25: `get_bm25_index().query(query, top_k=10)` (run in executor to avoid blocking)
- [ ] Merge results with `rrf_merge(vector_hits, bm25_hits)`
- [ ] If Phase 3 (re-ranking) is active, apply `rerank()` on the merged list as a final pass

### 4. Trigger BM25 index rebuild after ingestion
- [ ] In `backend/routers/podcast.py`, after ingestion completes (or in the Celery task callback if Phase 4.5 is active), call `bm25_index.rebuild()`

---

## Acceptance Criteria

- [ ] `rank-bm25` appears in `uv.lock` after `uv sync`
- [ ] `backend/agents/bm25_index.py` exists with `BM25Index`, `get_bm25_index()`, and `rrf_merge()`
- [ ] On server startup, `get_bm25_index()` successfully loads all chunks from ChromaDB without errors (check server logs)
- [ ] `POST /chat` with a query containing an exact term (e.g. a supplement name) returns at least one source that includes that exact term in its text
- [ ] `POST /chat` response time does not increase by more than 500 ms compared to the pre-Phase-7 baseline (BM25 is fast but index load should be profiled)
- [ ] `rrf_merge` produces no duplicate chunks in its output
- [ ] After ingesting a new episode, calling `rebuild()` causes the BM25 index to include the new episode's chunks (verified by querying for a term unique to that episode)
- [ ] `GET /chat/stream` (SSE) still streams correctly after the change

---

## Notes

- The BM25 index is built from all chunks in ChromaDB at startup — this could be slow (~1–2 s) for large corpora. Run it lazily (first request) or in a background startup event.
- ChromaDB `collection.get()` returns all documents; for very large collections (10k+ chunks) consider paginating with `limit` + `offset`.
- If Phase 3 re-ranking is active, the RRF merge feeds the re-ranker, which then applies the final precision pass. The pipeline becomes: vector search + BM25 → RRF merge → cross-encoder re-rank → top-3 chunks.
