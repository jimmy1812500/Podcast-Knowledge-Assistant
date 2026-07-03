"""
Retrieval node: embed query with BGE instruction prefix → query ChromaDB top-k.
"""

from __future__ import annotations

import asyncio

from backend.agents.reranker import rerank
from backend.agents.state import AgentState
from backend.etl.embeddings import COLLECTION_NAME, embed_query, get_chroma

TOP_K = 10


def _query_chroma_sync(vec: list[float]) -> list[dict]:
    chroma = get_chroma()
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    results = collection.query(
        query_embeddings=[vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    context = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        context.append(
            {
                "text": doc,
                "source": meta.get("source_file", ""),
                "timestamp_start": float(meta.get("timestamp_start", 0.0)),
                "timestamp_end": float(meta.get("timestamp_end", 0.0)),
                "speaker": meta.get("speaker", "UNKNOWN"),
                "score": round(1.0 - float(dist), 4),
            }
        )
    return context


async def retrieve_node(state: AgentState) -> dict:
    query = state.get("search_query") or state["query"]
    vec = await embed_query(query)
    loop = asyncio.get_event_loop()
    candidates = await loop.run_in_executor(None, _query_chroma_sync, vec)
    context = await loop.run_in_executor(None, rerank, query, candidates)
    return {
        "context": context,
        "iteration": state.get("iteration", 0) + 1,
    }
