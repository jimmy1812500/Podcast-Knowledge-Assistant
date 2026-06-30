# Phase 2 — RAG Chat Agent

## Goal
Answer natural-language questions about ingested podcast content using a local LLM (Ollama gemma3:4b), retrieval-augmented generation, and a self-correcting LangGraph agent loop. No cloud API required.

---

## Architecture

```
User Query
    │
    ▼
[retrieve]  embed query (BGE prefix) → ChromaDB top-5 cosine search
    │
    ▼
[evaluate]  gemma3:4b grades context: CONFIDENT / INSUFFICIENT_CONTEXT / AMBIGUOUS
    │
    ├─ CONFIDENT ────────────────────────────────────────────┐
    │                                                         │
    ├─ INSUFFICIENT_CONTEXT (iter < 3) → [expand]            │
    │       gemma3:4b rephrases query → back to [retrieve]   │
    │                                                         │
    └─ INSUFFICIENT_CONTEXT (iter ≥ 3) ─────────────────────┤
    └─ AMBIGUOUS ────────────────────────────────────────────┤
                                                             │
                                                             ▼
                                                       [synthesize]
                                                  gemma3:4b generates answer
                                                             │
                                                             ▼
                                                           END
```

---

## File Structure

```
backend/
├── agents/
│   ├── state.py        # AgentState TypedDict
│   ├── retrieval.py    # embed query → ChromaDB top-5
│   ├── evaluator.py    # gemma3:4b grades retrieved context
│   ├── synthesis.py    # gemma3:4b generates answer (batch + streaming)
│   └── graph.py        # LangGraph StateGraph, query expand node, routing
├── routers/
│   └── chat.py         # POST /chat  and  GET /chat/stream (SSE)
├── static/
│   └── index.html      # Chat UI (served at GET /)
└── main.py             # mounts chat router + static files
```

---

## Agent State

```python
class AgentState(TypedDict):
    query: str          # original user question (never mutated)
    search_query: str   # may be rephrased by expand node
    context: list[dict] # retrieved chunks from ChromaDB
    evaluation: str     # CONFIDENT | INSUFFICIENT_CONTEXT | AMBIGUOUS
    answer: str
    iteration: int      # retrieval attempt count (max 3)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Chat UI (browser) |
| POST | `/chat` | Synchronous JSON response |
| GET | `/chat/stream?q=...` | SSE streaming response |

### POST /chat — request / response

```json
// Request
{ "query": "What did Dr. Norton say about protein intake?" }

// Response
{
  "query": "...",
  "answer": "According to the podcast transcript...",
  "evaluation": "CONFIDENT",
  "sources": [
    {
      "source": "Essentials: The Science of Eating for Health...",
      "timestamp_start": 807.0,
      "timestamp_end": 830.0,
      "speaker": "UNKNOWN",
      "score": 0.4155,
      "text": "..."
    }
  ]
}
```

### GET /chat/stream — SSE event sequence

```
data: {"type": "sources", "sources": [...], "evaluation": "CONFIDENT"}

data: {"type": "token", "content": "According"}
data: {"type": "token", "content": " to"}
data: {"type": "token", "content": " the"}
...

data: {"type": "done"}
```

---

## Key Design Decisions

- **BGE query prefix** — queries are prefixed with `"Represent this sentence for searching relevant passages: "` before embedding. This matches the `bge-large-en-v1.5` model's recommended usage and improves retrieval accuracy over raw query embedding.
- **Evaluation before synthesis** — rather than always generating an answer, gemma3:4b first grades the retrieved context. If it's insufficient, the query is rephrased and retrieval retries (up to 3 times). This avoids hallucinated answers from irrelevant context.
- **Same LLM for all reasoning** — a single `gemma3:4b` instance (via Ollama) handles evaluation, query expansion, and synthesis. No external API, no cost per token.
- **Two endpoints for two UX patterns** — `POST /chat` runs the full LangGraph graph and returns a complete JSON blob; `GET /chat/stream` manually orchestrates the retrieve/evaluate/expand loop then streams synthesis tokens token-by-token via SSE for real-time display.
- **Singleton graph** — `build_graph()` is called once at startup and reused. LangGraph compilation is not free; recompiling on every request would add latency.

---

## Chat UI Features

Served at **http://localhost:8000**:

- Real-time token streaming as gemma3:4b generates the answer
- Confidence badge per response: green (CONFIDENT), yellow (AMBIGUOUS), red (Insufficient)
- Source cards below each answer showing episode title, timestamp range, and relevance score bar
- Enter key or Send button to submit

---

## Running

```bash
# Install dependencies (run once from project root)
uv sync

# Make sure Ollama is running with gemma3:4b
ollama serve
ollama pull gemma3:4b   # only needed on first run

# Start the API server (from project root)
uv run uvicorn backend.main:app --reload --port 8000

# Open chat UI
open http://localhost:8000

# Or use curl
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What foods help with fat loss?"}'

# Streaming
curl -N "http://localhost:8000/chat/stream?q=What+does+the+podcast+say+about+sleep?"
```

---

## Dependencies Added

```
ollama>=0.6.0      # Python client for local Ollama LLM
langgraph>=1.2.0   # Agent graph with conditional edges and state management
```
