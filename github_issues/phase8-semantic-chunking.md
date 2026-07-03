# Phase 8 — Semantic Chunking

## Background

The current text splitter in `backend/etl/splitter.py` creates fixed-size 500-character chunks with 100-character overlap, snapping to sentence boundaries. This is fast but splits text mechanically — a chunk can start mid-argument and end before the conclusion, weakening retrieval quality.

Semantic chunking groups consecutive sentences whose embeddings are similar, then splits where embedding similarity drops between adjacent sentences. The result is topic-coherent chunks that map better to the questions users ask.

**Scope:** new `backend/etl/semantic_splitter.py`; modification to `backend/etl/pipeline.py`. The existing `splitter.py` is kept as the default — semantic chunking is opt-in per ingestion run.

---

## Tasks

### 1. Add new dependencies
- [ ] Add `nltk>=3.9` to `pyproject.toml` under `[project] dependencies` (sentence tokenisation)
- [ ] Confirm `numpy` is already available (it is a transitive dependency of `sentence-transformers`); if not, add `numpy>=1.26`
- [ ] Run `uv sync` to lock

### 2. Create `backend/etl/semantic_splitter.py`

#### 2a. Helper: cosine similarity
- [ ] Implement `_cosine(a: list[float], b: list[float]) -> float` using `numpy` dot product

#### 2b. Download NLTK data on import
- [ ] Call `nltk.download("punkt_tab", quiet=True)` at module level so it auto-downloads on first use

#### 2c. Main function: `semantic_split`
- [ ] Signature: `semantic_split(text: str, breakpoint_threshold: float = 0.3, min_chunk_chars: int = 200) -> list[str]`
- [ ] Tokenise `text` into sentences using `nltk.sent_tokenize`
- [ ] If 1 or fewer sentences, return `[text]` immediately
- [ ] Embed all sentences in one batch call to `_encode_sync` (imported from `backend.etl.embeddings`)
- [ ] Compute cosine similarity between each adjacent pair of sentence embeddings
- [ ] Walk through sentences: when similarity drops below `breakpoint_threshold` **and** the current accumulated chunk is at least `min_chunk_chars`, close the chunk and start a new one
- [ ] Join sentences within each chunk with a single space and return the list of chunk strings

### 3. Modify `backend/etl/pipeline.py`
- [ ] Add a boolean parameter `use_semantic_chunking: bool = False` to the pipeline entry point (e.g. `run_pipeline()`)
- [ ] When `True`, call `semantic_split(full_text)` from `semantic_splitter` instead of `split_text` from `splitter`
- [ ] When `False` (default), keep existing behaviour — no regression for normal ingestion

### 4. Expose the flag in the CLI scripts
- [ ] In `backend/ingest_batch.py`, add a `--semantic-chunking` CLI flag that passes `use_semantic_chunking=True` to `run_pipeline`

---

## Acceptance Criteria

- [ ] `nltk` appears in `uv.lock` after `uv sync`
- [ ] `backend/etl/semantic_splitter.py` exists and `semantic_split("some long text...")` returns a non-empty list of strings without errors
- [ ] Running `uv run python backend/ingest_batch_1.py` (or equivalent single-episode test) with default settings still works — the existing fixed-size splitter is unchanged
- [ ] Running ingestion with `--semantic-chunking` completes without errors and stores chunks in ChromaDB
- [ ] Chunks produced by `semantic_split` are verifiably more coherent: inspect 5 random chunks and confirm none split mid-sentence or cut a thought abruptly
- [ ] Average chunk length with semantic splitting is ≥ 300 characters (not creating trivially small fragments)
- [ ] NLTK `punkt_tab` data downloads silently on first run and does not require manual setup
- [ ] No changes were made to `backend/etl/splitter.py` — the original fixed splitter is still intact and used by default

---

## Notes

- `_encode_sync` is called synchronously inside `semantic_split` — this is intentional since ingestion already runs outside the async event loop (in a Celery worker or CLI script). Do not call `embed_texts` (the async version) here.
- The `breakpoint_threshold=0.3` default was chosen empirically. Tune it per corpus: lower values (0.2) produce fewer, longer chunks; higher values (0.5) produce more, shorter chunks.
- Semantic chunking adds one full embedding pass over every sentence in the transcript, which is slower than fixed splitting. Expect ~30–60 extra seconds per 90-minute episode on CPU.
- The `Chunk` dataclass from `splitter.py` (with `char_start`/`char_end` for timestamp mapping) is not used by `semantic_split`. Timestamp mapping for semantic chunks can be added in a future issue.
