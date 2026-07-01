"""
Chat endpoints:
  POST /chat          — synchronous RAG answer (JSON)
  GET  /chat/stream   — SSE streaming answer (?q=<question>)
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agents.evaluator import evaluate_node
from backend.agents.graph import MAX_ITERATIONS, expand_query_node, get_graph
from backend.agents.retrieval import retrieve_node
from backend.agents.state import AgentState
from backend.agents.synthesis import stream_synthesis

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str


class SourceRef(BaseModel):
    source: str
    timestamp_start: float
    timestamp_end: float
    speaker: str
    score: float
    text: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    evaluation: str
    sources: list[SourceRef]


# ── POST /chat ────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    graph = get_graph()
    init_state: AgentState = {
        "query": req.query,
        "search_query": req.query,
        "context": [],
        "evaluation": "",
        "answer": "",
        "iteration": 0,
    }
    final = await graph.ainvoke(init_state)
    return ChatResponse(
        query=final["query"],
        answer=final["answer"],
        evaluation=final["evaluation"],
        sources=[SourceRef(**c) for c in final.get("context", [])],
    )


# ── GET /chat/stream ──────────────────────────────────────────────────────────

async def _sse_generator(query: str) -> AsyncIterator[str]:
    state: AgentState = {
        "query": query,
        "search_query": query,
        "context": [],
        "evaluation": "",
        "answer": "",
        "iteration": 0,
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
        }
        for c in state.get("context", [])
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'evaluation': state['evaluation']})}\n\n"

    # Stream synthesis tokens
    async for token in stream_synthesis(state):
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.get("/stream")
async def chat_stream(q: str) -> StreamingResponse:
    return StreamingResponse(
        _sse_generator(q),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
