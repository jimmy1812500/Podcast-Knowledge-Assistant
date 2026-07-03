# Phase 3 — Cross-Encoder Re-ranking

## Background

The current retrieval step in `backend/agents/retrieval.py` fetches the top-5 chunks by cosine similarity (BGE embeddings). Cosine similarity ranks by broad semantic closeness but cannot distinguish which chunk most precisely answers the query. A cross-encoder model reads the query and each candidate chunk together — producing a much more accurate relevance score at the cost of a small latency pass (~50 ms on CPU).

This is the single highest-leverage retrieval improvement because it requires no new infrastructure and sits entirely within the existing agent graph.

**Scope:** `backend/agents/` only. No changes to ETL, UI, or API contract.

---

## Tasks

### 1. Create `backend/agents/reranker.py`
- [ ] Add a `CrossEncoder` singleton loader using `cross-encoder/ms-marco-MiniLM-L-6-v2` (~85 MB, auto-downloaded from HuggingFace on first use)
- [ ] Implement `rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]`:
  - Build `(query, chunk["text"])` pairs for every candidate chunk
  - Call `model.predict(pairs)` to get float scores
  - Return the `top_k` chunks sorted by score descending

### 2. Modify `backend/agents/retrieval.py`
- [ ] Increase `TOP_K` from `5` to `10` so the re-ranker has more candidates to filter
- [ ] Import `rerank` from `backend.agents.reranker`
- [ ] After `_query_chroma_sync` returns the 10 candidates, call `rerank(query, context, top_k=3)` before returning the final context list

### 3. Smoke-test the pipeline
- [ ] Start the server: `uv run uvicorn backend.main:app --reload --port 8000`
- [ ] Send a test query via `POST /chat` and confirm the response still returns 3 sources
- [ ] Manually compare source relevance against the old top-5 cosine results to sanity-check improvement

---

## Acceptance Criteria

- [ ] `backend/agents/reranker.py` exists and contains a working `rerank()` function
- [ ] `TOP_K` in `retrieval.py` is set to `10`
- [ ] `rerank()` is called inside `retrieve_node` before the context is returned to the graph
- [ ] `POST /chat` still returns a valid `ChatResponse` with `sources` populated
- [ ] The cross-encoder model downloads and caches on first run without errors
- [ ] No changes were made to `graph.py`, `evaluator.py`, `synthesis.py`, or any ETL files
- [ ] `GET /chat/stream` (SSE) also works correctly end-to-end after the change

---

## Notes

- `sentence-transformers` is already installed — no new dependency needed
- Start with `ms-marco-MiniLM-L-6-v2` (L-6). Upgrade to `ms-marco-MiniLM-L-12-v2` only if you find L-6 accuracy insufficient (it is ~2× slower)
- The re-ranker is transparent to the LangGraph graph — `graph.py` does not need to change
