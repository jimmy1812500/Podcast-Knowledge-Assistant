"""
Retrieval node: embed query with BGE instruction prefix → query ChromaDB top-k.
"""

from __future__ import annotations

import asyncio

from backend.agents.reranker import rerank
from backend.agents.state import AgentState
from backend.etl.embeddings import COLLECTION_NAME, embed_query, get_chroma

TOP_K = 10

# Maps name/role keywords (lowercase) → ChromaDB speaker label.
# Update this when new episodes with different guests are ingested.
# SPEAKER_02 = Andrew Huberman (host)
# SPEAKER_01 = Dr. Layne Norton (guest, episode: Essentials - Eating for Health)
# SPEAKER_00 = Ads / end-of-episode noise — never a useful source
SPEAKER_ALIASES: dict[str, str] = {
    "huberman": "SPEAKER_02",
    "andrew": "SPEAKER_02",
    "host": "SPEAKER_02",
    "norton": "SPEAKER_01",
    "layne": "SPEAKER_01",
    "guest": "SPEAKER_01",
    "dr. norton": "SPEAKER_01",
}

# These speaker labels contain only ads or noise — always excluded
_AD_SPEAKERS: set[str] = {"SPEAKER_00"}


def _detect_speaker_filter(query: str, messages: list | None = None) -> str | None:
    """Return the speaker label implied by the query or any prior human turn.

    Checks the current query first, then walks conversation history newest-first
    so that pronoun follow-ups ("his opinion") inherit the speaker from the
    turn where the name was first mentioned.
    """

    def _match(text: str) -> str | None:
        t = text.lower()
        for keyword, speaker in SPEAKER_ALIASES.items():
            if keyword in t:
                return speaker
        return None

    result = _match(query)
    if result:
        return result

    # Scan prior human messages (most recent first) for a speaker reference
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") == "human":
            result = _match(msg.content)
            if result:
                return result

    return None


def _query_chroma_sync(
    vec: list[float], user_id: str, podcast_filter: str | None = None
) -> list[dict]:
    chroma = get_chroma()
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if podcast_filter is not None:
        where: dict = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"podcast_id": {"$eq": podcast_filter}},
            ]
        }
    else:
        where = {"user_id": {"$eq": user_id}}
    results = collection.query(
        query_embeddings=[vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
        where=where,
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
                "audio_file": meta.get("audio_file", ""),
            }
        )
    return context


async def retrieve_node(state: AgentState) -> dict:
    query = state.get("search_query") or state["query"]
    user_id = state.get("user_id", "public")
    podcast_filter = state.get("podcast_filter")
    vec = await embed_query(query)
    loop = asyncio.get_running_loop()
    candidates = await loop.run_in_executor(None, _query_chroma_sync, vec, user_id, podcast_filter)

    # Detect speaker intent from the original query and conversation history.
    # Using state["query"] (not search_query) so rephrasing doesn't drop names.
    # Passing messages so pronoun follow-ups ("his opinion") resolve to the
    # speaker named in an earlier turn.
    speaker_filter = _detect_speaker_filter(state["query"], state.get("messages", []))

    if speaker_filter:
        # Query names a specific person — return only their chunks
        filtered = [c for c in candidates if c.get("speaker") == speaker_filter]
    else:
        # No specific person — exclude ad/noise chunks, allow all content speakers
        filtered = [c for c in candidates if c.get("speaker") not in _AD_SPEAKERS]

    # Fall back to non-ad candidates if the speaker filter removes everything,
    # so ad chunks never surface as a named speaker's answer.
    non_ad = [c for c in candidates if c.get("speaker") not in _AD_SPEAKERS]
    context = await loop.run_in_executor(None, rerank, query, filtered or non_ad or candidates)
    return {
        "context": context,
        "iteration": state.get("iteration", 0) + 1,
    }
