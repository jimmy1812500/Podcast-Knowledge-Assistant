"""
LangGraph RAG agent graph.

Flow:
  retrieve → evaluate → (CONFIDENT) → synthesize → END
                      → (INSUFFICIENT_CONTEXT, iter < MAX) → expand → retrieve → ...
                      → (INSUFFICIENT_CONTEXT, iter >= MAX) → synthesize → END
                      → (AMBIGUOUS) → synthesize → END
"""

from __future__ import annotations

import ollama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.agents.evaluator import OLLAMA_MODEL, evaluate_node
from backend.agents.retrieval import retrieve_node
from backend.agents.state import AgentState
from backend.agents.synthesis import synthesize_node

_checkpointer = MemorySaver()

MAX_ITERATIONS = 3


async def expand_query_node(state: AgentState) -> dict:
    """Ask Ollama to rephrase the query to find different relevant content."""
    prompt = (
        "Rephrase this question using different keywords to search a podcast transcript database:\n"
        f"{state['query']}\n\n"
        "Provide only the rephrased question, nothing else."
    )
    client = ollama.AsyncClient()
    resp = await client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"search_query": resp.message.content.strip()}


def _route_after_eval(state: AgentState) -> str:
    if state["evaluation"] == "CONFIDENT":
        return "synthesize"
    if state.get("iteration", 0) < MAX_ITERATIONS:
        return "expand"
    return "synthesize"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("expand", expand_query_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        _route_after_eval,
        {"synthesize": "synthesize", "expand": "expand"},
    )
    g.add_edge("expand", "retrieve")
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=_checkpointer)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
