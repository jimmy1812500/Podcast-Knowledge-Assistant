"""
Synthesis node: Ollama gemma3:4b generates the final answer.
Supports both batch (for /chat) and streaming (for /chat/stream SSE).
"""

from __future__ import annotations

from typing import AsyncIterator

import ollama

from backend.agents.state import AgentState

OLLAMA_MODEL = "gemma3:4b"

_SYNTH_PROMPT = """\
You are a knowledgeable assistant answering questions about the Huberman Lab podcast.

Question: {query}

Relevant excerpts from podcast transcripts:
{context}

Answer the question based on these excerpts. Include approximate timestamps or \
episode references when available. If the information is limited, say so clearly."""


def _build_context_text(context: list[dict]) -> str:
    parts = []
    for c in context:
        ts = f"{c['timestamp_start']:.0f}s–{c['timestamp_end']:.0f}s"
        parts.append(f"[{ts} | {c['source']}]\n{c['text']}")
    return "\n\n".join(parts)


async def synthesize_node(state: AgentState) -> dict:
    context_text = _build_context_text(state.get("context", []))
    prompt = _SYNTH_PROMPT.format(
        query=state["query"],
        context=context_text or "No relevant context was found.",
    )

    client = ollama.AsyncClient()
    resp = await client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"answer": resp.message.content.strip()}


async def stream_synthesis(state: AgentState) -> AsyncIterator[str]:
    """Yields answer tokens one at a time for SSE streaming."""
    context_text = _build_context_text(state.get("context", []))
    prompt = _SYNTH_PROMPT.format(
        query=state["query"],
        context=context_text or "No relevant context was found.",
    )

    client = ollama.AsyncClient()
    async for chunk in await client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        if chunk.message and chunk.message.content:
            yield chunk.message.content
