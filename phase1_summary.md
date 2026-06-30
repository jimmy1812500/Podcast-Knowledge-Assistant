# Phase 1 — ETL Pipeline

## Goal
Ingest Huberman Lab podcast episodes from RSS, transcribe the audio locally, and store searchable text chunks in ChromaDB — with no external AI API required.

---

## Architecture

```
RSS Feed → Download MP3 → Whisper ASR → Speaker Diarization → Text Splitter → Embeddings → ChromaDB
```

### Key components

| Component | Technology | Details |
|-----------|-----------|---------|
| RSS parsing | `feedparser` | Huberman Lab feed: `feeds.megaphone.fm/hubermanlab` |
| Audio download | `httpx` (async) | Concurrent downloads (2 at a time), resume-safe |
| Transcription | `faster-whisper` | `base` model, CPU, int8 — ~3–4 min per 90 min episode |
| Speaker diarization | `pyannote.audio 3.1` | Optional, requires HF_TOKEN with gated model access |
| Text chunking | Custom splitter | 500-char chunks, 100-char overlap, snaps to sentence boundaries |
| Embeddings | `sentence-transformers` | `BAAI/bge-large-en-v1.5` — 1024-dim, local, no API key |
| Vector store | `ChromaDB` | Cosine similarity (HNSW), persistent on disk at `./chroma_data` |

---

## File Structure

```
backend/
├── etl/
│   ├── podcast_rss.py     # RSS fetch + async MP3 download
│   ├── asr.py             # faster-whisper wrapper (singleton model cache)
│   ├── diarize.py         # pyannote diarization (optional)
│   ├── splitter.py        # overlapping text chunker
│   ├── embeddings.py      # BAAI/bge-large-en-v1.5 embed + ChromaDB store
│   └── pipeline.py        # orchestrates ASR → diarize → chunk → embed → store
├── routers/
│   ├── upload.py          # POST /upload  (multipart audio file)
│   └── podcast.py         # POST /ingest/podcast  (background RSS job)
├── ingest_batch.py        # CLI — batch ingest all 2025–2026 episodes
└── ingest_batch_1.py      # CLI — single-episode test (diarization disabled)
```

---

## ChromaDB Schema

Each stored chunk has:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | SHA-1 of `source:char_start` |
| `text` | string | Raw transcript text (~500 chars) |
| `source_file` | string | Episode title (used as resume key) |
| `timestamp_start` | float | Start time in seconds (from Whisper) |
| `timestamp_end` | float | End time in seconds |
| `confidence_score` | float | Average Whisper log-probability, mapped to [0, 1] |
| `speaker` | string | `SPEAKER_00`, `SPEAKER_01`, … or `UNKNOWN` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload an audio file directly (multipart, max 500 MB) |
| POST | `/ingest/podcast` | Start background RSS ingestion job |
| GET | `/ingest/status/{job_id}` | Poll job progress |

---

## CLI Usage

```bash
# Install dependencies (run once from project root)
uv sync

# Dry run — list episodes without downloading
uv run python backend/ingest_batch.py --dry-run

# Ingest all 2025–2026 episodes (resume-safe)
uv run python backend/ingest_batch.py --since 2025-01-01

# Ingest a date range with a larger Whisper model
uv run python backend/ingest_batch.py --since 2025-06-01 --until 2025-12-31 --model small

# Single-episode test (diarization off, full episode)
uv run python backend/ingest_batch_1.py
```

---

## Configuration

```bash
# .env
HF_TOKEN=hf_...          # Required only for speaker diarization
CHROMA_HOST=localhost     # Optional: connect to ChromaDB HTTP server
CHROMA_PORT=8001          # Falls back to ./chroma_data on disk if unreachable
```

---

## Key Design Decisions

- **Sequential transcription + diarization** — both are CPU-bound (CTranslate2 + PyTorch). Running in parallel causes core contention and was 3× slower in practice. Whisper runs first, then diarization.
- **Resume-safe ingestion** — before processing each episode, a ChromaDB `where` filter checks if any chunk with that `source_file` already exists. Already-indexed episodes are skipped.
- **No external AI API** — all inference (transcription, embedding) runs locally. Only `HF_TOKEN` is needed for the optional diarization model download.
- **BGE query prefix** — when embedding for retrieval (Phase 2), the prefix `"Represent this sentence for searching relevant passages: "` is prepended to improve cosine similarity results with `bge-large-en-v1.5`.

---

## Confirmed Working

- Ingested episode: **"Essentials: The Science of Eating for Health, Fat Loss & Lean Muscle | Dr. Layne Norton"**
- Chunks stored: **209**
- Transcription time: **~3m 41s** for a full ~90 min episode (Whisper base, CPU)
- Speaker diarization: disabled (pyannote gated model requires license acceptance at huggingface.co)
