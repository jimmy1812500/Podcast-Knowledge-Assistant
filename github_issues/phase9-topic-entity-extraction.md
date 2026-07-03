# Phase 9 — Topic & Entity Extraction (Knowledge Layer)

## Background

All chunks stored today contain only raw transcript text and timing metadata. This is sufficient for free-text semantic search but not for structured queries like:

- "Which episodes mention Andrew Huberman discussing magnesium?"
- "List all topics covered across ingested episodes"
- "What claims were made about protein timing?"

This phase adds a knowledge extraction pass during ingestion: for each chunk, `gemma3:4b` (via Ollama) extracts topics, named entities, and key claims. These are stored as ChromaDB metadata, enabling both pre-filtering at retrieval time and new browsing endpoints.

**Scope:** new `backend/etl/extractor.py`; modifications to `backend/etl/pipeline.py`, `backend/etl/embeddings.py`, `backend/routers/`; new `GET /topics` endpoint.

---

## Tasks

### 1. Create `backend/etl/extractor.py`

#### 1a. Extraction prompt
- [ ] Define a prompt that instructs `gemma3:4b` to extract three fields from a transcript excerpt:
  - `topics`: list of 1–3 word topic tags (e.g. `"sleep"`, `"cortisol"`, `"protein intake"`)
  - `entities`: named people, supplements, studies, or tools mentioned
  - `claims`: short factual claims made (max 3, one sentence each)
- [ ] Instruct the model to return **only** valid JSON: `{"topics": [...], "entities": [...], "claims": [...]}`

#### 1b. `extract_metadata` async function
- [ ] Signature: `async def extract_metadata(text: str) -> dict`
- [ ] Call `ollama.AsyncClient().chat(model="gemma3:4b", ..., format="json")`
- [ ] Parse the response with `json.loads()`; on `JSONDecodeError` return `{"topics": [], "entities": [], "claims": []}`

### 2. Modify `backend/etl/pipeline.py`
- [ ] Add a boolean parameter `extract_knowledge: bool = False` to `run_pipeline()`
- [ ] When `True`, after chunking and before calling `store_chunks()`, run `extract_metadata(chunk.text)` for each chunk
  - Use `asyncio.gather` to run extractions concurrently (batch size of 5 to avoid overwhelming Ollama)
- [ ] Attach the extracted `topics`, `entities`, `claims` lists to each chunk dict

### 3. Modify `backend/etl/embeddings.py` — `store_chunks()`
- [ ] Accept optional `topics: list[str]`, `entities: list[str]`, `claims: list[str]` per chunk
- [ ] Join each list as a comma-separated string (ChromaDB metadata values must be scalar strings)
- [ ] Store them as `"topics"`, `"entities"`, `"claims"` in the ChromaDB metadata dict

### 4. Expose a `GET /topics` endpoint

#### 4a. New router `backend/routers/topics.py`
- [ ] Query the ChromaDB collection with `collection.get(include=["metadatas"])`
- [ ] Aggregate all `topics` values: split each comma-separated string, count occurrences
- [ ] Return a sorted list of `{"topic": str, "count": int}` objects

#### 4b. Register the router in `backend/main.py`
- [ ] `app.include_router(topics.router)`

### 5. Use topics as a retrieval pre-filter in `backend/agents/retrieval.py`
- [ ] Before the vector search, scan the user query for known topic tags (load the tag list from the `GET /topics` aggregation or a cached set)
- [ ] If a match is found, add a ChromaDB `where={"topics": {"$contains": matched_tag}}` pre-filter to narrow the candidate set

---

## Acceptance Criteria

- [ ] `backend/etl/extractor.py` exists and `extract_metadata("some transcript text")` returns a dict with `topics`, `entities`, `claims` keys
- [ ] Running ingestion with `extract_knowledge=True` stores `topics`, `entities`, `claims` in ChromaDB metadata for every chunk (verified via `collection.get()`)
- [ ] Running ingestion with default settings (`extract_knowledge=False`) completes without calling `extract_metadata` — no regression on ingestion speed
- [ ] `GET /topics` returns a non-empty JSON array after at least one episode has been ingested with extraction enabled
- [ ] `GET /topics` response includes at least 5 distinct topic tags for a typical 90-minute Huberman Lab episode
- [ ] `POST /chat` with a query matching a known topic tag returns sources pre-filtered to chunks that contain that topic tag in their metadata
- [ ] On `JSONDecodeError` from the LLM, `extract_metadata` returns the safe fallback `{"topics": [], "entities": [], "claims": []}` without crashing the ingestion pipeline
- [ ] The extraction LLM pass adds no more than 5 minutes to the ingestion time of a single 90-minute episode (with `concurrency=5` gather batching)

---

## Notes

- Knowledge extraction is intentionally opt-in (`extract_knowledge=False` default) because it adds significant ingestion time. Mark episodes that have been extracted in their metadata (e.g. `"extracted": "true"`) so you can skip re-extraction on re-ingestion.
- ChromaDB `$contains` filter on a string field does a substring match — ensure topic tags are stored lowercase and stripped of whitespace to avoid match failures.
- If Phase 4.5 (Celery) is active, pass `extract_knowledge` as a Celery task argument so it can be toggled per job from the API.
- Claims are the most valuable field for a future "fact-checking" or "citation" feature — store them faithfully even if the UI doesn't display them yet.
