# Podcast Knowledge Assistant

A local-first RAG (Retrieval-Augmented Generation) assistant that ingests podcast episodes, transcribes and diarizes them, and answers natural-language questions about their content — with source timestamps and speaker attribution. Everything runs on-device: transcription (Whisper), embeddings (BGE), and reasoning (Ollama) all execute locally with no external AI API required.

## Features

- **Automated ingestion** — pulls episodes straight from an RSS feed (currently configured for Huberman Lab), or accepts a direct audio file upload
- **Local transcription** — `faster-whisper` converts audio to text on CPU, no cloud API needed
- **Speaker diarization** — optional `pyannote.audio` integration labels who said what
- **Semantic search** — transcripts are chunked, embedded with `BAAI/bge-large-en-v1.5`, and stored in ChromaDB for cosine-similarity retrieval
- **Self-correcting RAG agent** — a LangGraph agent retrieves context, grades its own confidence, rephrases and retries the query when context is insufficient, then synthesizes an answer with a local LLM (Ollama `gemma3:4b`)
- **Streaming chat UI** — built-in web UI with real-time token streaming, confidence badges, and source cards showing episode, timestamp, and relevance score
- **Resume-safe batch ingestion** — CLI tool to bulk-ingest a date range of episodes, skipping ones already indexed

## Getting Started

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Docker](https://www.docker.com/) (optional, for running ChromaDB as a server instead of on-disk)
- [Ollama](https://ollama.com/) running locally with the `gemma3:4b` model, for the chat agent
- A [Hugging Face token](https://huggingface.co/settings/tokens) with access accepted for [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) (optional — only needed for speaker diarization; without it, speakers are labeled `UNKNOWN`)

### Installation

```bash
git clone <repository-url>
cd Podcast-Knowledge-Assistant
uv sync
```

Copy the example environment file and fill in your values:

```bash
cp backend/.env.example backend/.env
```

| Variable | Required | Description |
|----------|----------|--------------|
| `HF_TOKEN` | No | Hugging Face token for speaker diarization. Leave blank to skip (speakers show as `UNKNOWN`). |
| `CHROMA_HOST` / `CHROMA_PORT` | No | Connect to a ChromaDB server. Falls back to on-disk storage at `./chroma_data` if unreachable. |

(Optional) Start a ChromaDB server via Docker Compose instead of using on-disk storage:

```bash
make chroma-up
```

(Optional) Pull the local LLM used by the chat agent:

```bash
ollama pull gemma3:4b
```

## Usage

Start the API server:

```bash
make run
# or: uv run uvicorn backend.main:app --reload --port 8000
```

Open the chat UI in your browser:

```bash
open http://localhost:8000
```

### Ingesting podcast episodes

Batch-ingest from RSS (resume-safe — already-indexed episodes are skipped):

```bash
# Dry run — list episodes without downloading
make ingest ARGS="--dry-run"

# Ingest all episodes since a given date
make ingest ARGS="--since 2025-01-01"

# Use a larger Whisper model for better accuracy
make ingest ARGS="--since 2025-06-01 --until 2025-12-31 --model small"
```

Or upload a single audio file directly via `POST /upload` (multipart, max 500 MB).

### Querying via API

```bash
# Synchronous response
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What did the guest say about protein intake?"}'

# Streaming response (SSE)
curl -N "http://localhost:8000/chat/stream?q=What+does+the+episode+say+about+sleep?"
```

### Development commands

```bash
make lint              # check code style with ruff
make format             # auto-format code with ruff
make test               # run the test suite
make precommit-install  # install git pre-commit hooks
make chroma-clean       # wipe all ingested chunks from ChromaDB
```

## Project Structure

```
backend/
├── agents/          # LangGraph RAG agent (retrieve → evaluate → expand → synthesize)
├── etl/             # RSS fetch, transcription, diarization, chunking, embeddings
├── routers/         # FastAPI routes: /upload, /ingest/podcast, /chat
├── static/          # Chat UI (served at GET /)
├── ingest_batch.py  # CLI for resume-safe batch RSS ingestion
└── main.py          # FastAPI app entrypoint
```

See [phase1_summary.md](phase1_summary.md) for ETL pipeline details and [phase2_summary.md](phase2_summary.md) for the RAG agent architecture.
