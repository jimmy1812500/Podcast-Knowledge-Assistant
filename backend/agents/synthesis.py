"""
Synthesis node: Ollama gemma3:4b generates the final answer.
Supports both batch (for /chat) and streaming (for /chat/stream SSE).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import ollama
from langchain_core.messages import AIMessage

from backend.agents.state import AgentState

OLLAMA_MODEL = "gemma3:4b"

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant answering questions about the Huberman Lab podcast."
)

_USER_PROMPT = """\
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


def _build_ollama_messages(state: AgentState, context_text: str) -> list[dict]:
    """Build the message list for Ollama, including prior conversation history."""
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject prior turns (all messages except the current HumanMessage at the end)
    prior = state.get("messages", [])[:-1]
    for msg in prior:
        role = "user" if getattr(msg, "type", "") == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})

    # Current question with retrieved context
    messages.append(
        {
            "role": "user",
            "content": _USER_PROMPT.format(
                query=state["query"],
                context=context_text or "No relevant context was found.",
            ),
        }
    )
    return messages


async def synthesize_node(state: AgentState) -> dict:
    context_text = _build_context_text(state.get("context", []))
    messages = _build_ollama_messages(state, context_text)

    client = ollama.AsyncClient()
    resp = await client.chat(model=OLLAMA_MODEL, messages=messages)
    answer = resp.message.content.strip()
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


async def stream_synthesis(state: AgentState) -> AsyncIterator[str]:
    """Yields answer tokens one at a time for SSE streaming."""
    context_text = _build_context_text(state.get("context", []))
    messages = _build_ollama_messages(state, context_text)

    client = ollama.AsyncClient()
    async for chunk in await client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True,
    ):
        if chunk.message and chunk.message.content:
            yield chunk.message.content
