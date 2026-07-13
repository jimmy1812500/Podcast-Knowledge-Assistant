"""
Embeddings model + ChromaDB vector store.

Embeds text chunks locally with sentence-transformers (BAAI/bge-large-en-v1.5).
No API key required. Model (~1.3 GB) is downloaded from HuggingFace on first use
and cached in ~/.cache/huggingface/hub/.

Optional env vars:
    CHROMA_HOST  (default localhost)
    CHROMA_PORT  (default 8001)
Falls back to a local PersistentClient when the ChromaDB server is unreachable.

Embedding dimension: 1024 (bge-large-en-v1.5).
Note: switching models changes vector dimensions — drop and recreate the
ChromaDB collection if you previously ran with a different model.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "transcription_chunks"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024
_ENCODE_BATCH = 32  # chunks per SentenceTransformer encode call


def _make_chroma() -> chromadb.ClientAPI:
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8001"))
    try:
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
        return client
    except Exception:
        return chromadb.PersistentClient(path="./backend/chroma_data")


_chroma: chromadb.ClientAPI | None = None
_model: SentenceTransformer | None = None


def get_chroma() -> chromadb.ClientAPI:
    global _chroma
    if _chroma is None:
        _chroma = _make_chroma()
    return _chroma


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _encode_sync(texts: list[str]) -> list[list[float]]:
    model = _load_model()
    # normalize_embeddings=True → unit-length vectors, optimal for cosine similarity
    vecs = model.encode(
        texts,
        batch_size=_ENCODE_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.tolist()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts locally using BAAI/bge-large-en-v1.5."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_sync, texts)


# BGE instruction prefix for query embedding (improves retrieval accuracy)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _encode_query_sync(query: str) -> list[float]:
    return _encode_sync([_BGE_QUERY_PREFIX + query])[0]


async def embed_query(query: str) -> list[float]:
    """Embed a single search query with the BGE query instruction prefix."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_query_sync, query)


async def store_chunks(
    chunks_with_meta: list[dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
    user_id: str = "public",
) -> int:
    """
    Embed and upsert chunks into ChromaDB.

    Each dict must contain:
        chunk_id (str), text (str), source_file (str),
        timestamp_start (float), timestamp_end (float),
        confidence_score (float), speaker (str),
        podcast_id (str)  — optional, defaults to "unknown" if absent

    Returns the number of chunks stored.
    """
    if not chunks_with_meta:
        return 0

    texts = [c["text"] for c in chunks_with_meta]
    embeddings = await embed_texts(texts)

    chroma = get_chroma()
    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[c["chunk_id"] for c in chunks_with_meta],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source_file": c["source_file"],
                "timestamp_start": c["timestamp_start"],
                "timestamp_end": c["timestamp_end"],
                "confidence_score": c["confidence_score"],
                "speaker": c.get("speaker", "UNKNOWN"),
                "user_id": user_id,
                "podcast_id": c.get("podcast_id", "unknown"),
            }
            for c in chunks_with_meta
        ],
    )

    return len(chunks_with_meta)
