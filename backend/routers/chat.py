"""
Chat endpoints:
  POST /chat          — synchronous RAG answer (JSON)
  GET  /chat/stream   — SSE streaming answer (?q=<question>&session_id=<id>)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from backend.agents.evaluator import evaluate_node
from backend.agents.graph import MAX_ITERATIONS, expand_query_node, get_graph
from backend.agents.retrieval import retrieve_node
from backend.agents.state import AgentState
from backend.agents.synthesis import stream_synthesis
from backend.etl.podcast_registry import PODCASTS

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory conversation store for SSE streaming sessions
_streaming_sessions: dict[str, list] = {}


# ── Request / response models ─────────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    podcast_filter: str | None = None


class SourceRef(BaseModel):
    source: str
    timestamp_start: float
    timestamp_end: float
    speaker: str
    score: float
    text: str
    audio_file: str = ""


class ChatResponse(BaseModel):
    query: str
    answer: str
    evaluation: str
    sources: list[SourceRef]


# ── POST /chat ────────────────────────────────────────────────────────────────


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    x_user_id: str | None = Header(None, alias="X-User-ID"),
) -> ChatResponse:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header is required.")

    # Validate against the registry only for short registry-key style values.
    # URL-based filters (ingested via feed_url) contain "://" and must pass through,
    # because ChromaDB stores the URL itself as podcast_id for those chunks.
    if (
        req.podcast_filter is not None
        and "://" not in req.podcast_filter
        and req.podcast_filter not in PODCASTS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown podcast_filter {req.podcast_filter!r}. Valid IDs: {list(PODCASTS)}",
        )

    graph = get_graph()
    init_state: AgentState = {
        "query": req.query,
        "search_query": req.query,
        "context": [],
        "evaluation": "",
        "answer": "",
        "iteration": 0,
        "messages": [HumanMessage(content=req.query)],
        "user_id": x_user_id,
        "podcast_filter": req.podcast_filter,
    }
    config = {"configurable": {"thread_id": f"{x_user_id}:{req.session_id or 'default'}"}}
    final = await graph.ainvoke(init_state, config=config)
    return ChatResponse(
        query=final["query"],
        answer=final["answer"],
        evaluation=final["evaluation"],
        sources=[SourceRef(**c) for c in final.get("context", [])],
    )


# ── GET /chat/stream ──────────────────────────────────────────────────────────


async def _sse_generator(
    query: str, session_id: str, user_id: str, podcast_filter: str | None = None
) -> AsyncIterator[str]:
    session_key = f"{user_id}:{session_id}"
    prior_messages = _streaming_sessions.get(session_key, [])

    state: AgentState = {
        "query": query,
        "search_query": query,
        "context": [],
        "evaluation": "",
        "answer": "",
        "iteration": 0,
        "messages": prior_messages + [HumanMessage(content=query)],
        "user_id": user_id,
        "podcast_filter": podcast_filter,
    }

    # Run retrieve → evaluate → expand loop (without synthesis)
    while state["iteration"] < MAX_ITERATIONS:
        updates = await retrieve_node(state)
        state = {**state, **updates}

        updates = await evaluate_node(state)
        state = {**state, **updates}

        if state["evaluation"] != "INSUFFICIENT_CONTEXT":
            break

        updates = await expand_query_node(state)
        state = {**state, **updates}

    # Send retrieved sources and evaluation result
    sources = [
        {
            "source": c["source"],
            "timestamp_start": c["timestamp_start"],
            "timestamp_end": c["timestamp_end"],
            "speaker": c["speaker"],
            "score": c["score"],
            "audio_file": c.get("audio_file", ""),
        }
        for c in state.get("context", [])
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'evaluation': state['evaluation']})}\n\n"

    # Stream synthesis tokens and accumulate full answer
    full_answer = ""
    async for token in stream_synthesis(state):
        full_answer += token
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    # Persist updated conversation history for this session
    _streaming_sessions[session_key] = prior_messages + [
        HumanMessage(content=query),
        AIMessage(content=full_answer),
    ]

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.get("/stream")
async def chat_stream(
    q: str,
    session_id: str | None = None,
    podcast: str | None = None,
    x_user_id: str | None = Header(None, alias="X-User-ID"),
) -> StreamingResponse:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header is required.")

    if podcast is not None and "://" not in podcast and podcast not in PODCASTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown podcast {podcast!r}. Valid IDs: {list(PODCASTS)}",
        )

    return StreamingResponse(
        _sse_generator(q, session_id or "default", x_user_id, podcast),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
