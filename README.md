# Podcast Knowledge Assistant

Ask natural-language questions about podcast episodes and get cited, timestamped answers — all running locally with no cloud APIs required.

Most podcast knowledge is trapped in audio. This project transcribes episodes, chunks the transcripts, and stores them in a vector database so you can query them conversationally. The RAG agent retrieves relevant passages, re-ranks them for precision, and generates an answer with source attribution showing the exact timestamp in the episode.

---

## Features

- **Audio ingestion** — upload MP3 files directly or pull episodes from an RSS feed URL
- **Whisper transcription** — local ASR via `faster-whisper`; no audio ever leaves your machine
- **Speaker diarization** — optional per-speaker labels using `pyannote-audio` (requires a Hugging Face token)
- **Vector search** — chunks are embedded and stored in ChromaDB for semantic retrieval
- **Cross-encoder re-ranking** — a `ms-marco-MiniLM-L-6-v2` cross-encoder rescores the top candidates for higher precision before synthesis
- **LangGraph RAG agent** — retrieve → evaluate → expand query loop (up to 3 iterations) before synthesizing a final answer
- **Multi-turn memory** — `MemorySaver` checkpointing lets you ask follow-up questions within a session
- **SSE streaming** — the `/chat/stream` endpoint streams tokens as they are generated
- **Local LLM** — uses Ollama (`gemma3:4b` by default); swap the model in `backend/agents/evaluator.py`

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| [uv](https://github.com/astral-sh/uv) | latest |
| [Ollama](https://ollama.com) | latest |
| Hugging Face token | optional — only needed for speaker diarization |

Pull the LLM before starting:

```bash
ollama pull gemma3:4b
```

---

## Quick Start

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd "Podcast Knowledge Assistant"
uv sync

# 2. Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env — add HF_TOKEN if you want speaker labels

# 3. Start the API server
uv run uvicorn backend.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) to use the web UI.

### Ingest your first episode

**Via the web UI** — use the upload form on the homepage.

**Via the API directly:**

```bash
# Upload an audio file
curl -X POST http://localhost:8000/upload \
  -F "file=@episode.mp3"

# Or pull from an RSS feed
curl -X POST http://localhost:8000/ingest/podcast \
  -H "Content-Type: application/json" \
  -d '{"rss_url": "https://feeds.megaphone.fm/hubermanlab"}'
```

### Query the knowledge base

```bash
# Single-turn JSON response
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What did the guest say about sleep and cortisol?", "session_id": "my-session"}'

# Streaming response (SSE)
curl "http://localhost:8000/chat/stream?q=What+is+the+best+time+to+exercise?&session_id=my-session"
```

The response includes `answer`, `sources` (with `source`, `timestamp_start`, `timestamp_end`, `speaker`, and the matched `text`), and an `evaluation` field (`CONFIDENT` | `INSUFFICIENT_CONTEXT` | `AMBIGUOUS`).

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | No | Hugging Face token — enables speaker diarization via `pyannote-audio`. Skip this to label all speakers as `UNKNOWN`. |
| `CHROMA_HOST` | No | ChromaDB host when running via Docker. Defaults to local disk at `./chroma_data`. |
| `CHROMA_PORT` | No | ChromaDB port. Defaults to `8001`. |

---

## Project Structure

```
backend/
  etl/
    asr.py          # Whisper transcription (faster-whisper)
    diarize.py      # Speaker diarization (pyannote-audio)
    splitter.py     # Overlapping text chunker
    embeddings.py   # Chunk embedding + ChromaDB upsert
    pipeline.py     # ETL orchestrator: audio → chunks → vector store
    podcast_rss.py  # RSS feed episode downloader
  agents/
    state.py        # LangGraph AgentState (with multi-turn message history)
    graph.py        # RAG agent graph: retrieve → evaluate → expand → synthesize
    retrieval.py    # ChromaDB query + cross-encoder re-ranking
    reranker.py     # Cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2)
    evaluator.py    # LLM-based context quality evaluator
    synthesis.py    # Answer generation (streaming + non-streaming)
  routers/
    chat.py         # POST /chat, GET /chat/stream
    upload.py       # POST /upload
    podcast.py      # POST /ingest/podcast
  static/
    index.html      # Single-page web UI
  main.py           # FastAPI app entry point
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| Audio transcription | faster-whisper (CTranslate2 backend) |
| Speaker diarization | pyannote-audio |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (BGE model) |
| Re-ranking | sentence-transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| Agent orchestration | LangGraph |
| Local LLM | Ollama (`gemma3:4b`) |
| Package manager | uv |
