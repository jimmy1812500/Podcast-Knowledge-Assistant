"""
Evaluator node: Ollama gemma3:4b grades whether retrieved context is sufficient.
Returns: CONFIDENT | INSUFFICIENT_CONTEXT | AMBIGUOUS
"""

from __future__ import annotations

import ollama

from agents.state import AgentState

OLLAMA_MODEL = "gemma3:4b"

_EVAL_PROMPT = """\
You evaluate whether retrieved context can answer a question.

Question: {query}

Retrieved context:
{context}

Reply with exactly one word:
- CONFIDENT            — context clearly and directly answers the question
- INSUFFICIENT_CONTEXT — context lacks relevant information to answer
- AMBIGUOUS            — context is related but the answer is unclear"""


async def evaluate_node(state: AgentState) -> dict:
    if not state.get("context"):
        return {"evaluation": "INSUFFICIENT_CONTEXT"}

    context_text = "\n\n".join(
        f"[{i + 1}] {c['text']}" for i, c in enumerate(state["context"])
    )
    prompt = _EVAL_PROMPT.format(query=state["query"], context=context_text)

    client = ollama.AsyncClient()
    resp = await client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.message.content.strip().upper()

    if "CONFIDENT" in raw:
        evaluation = "CONFIDENT"
    elif "INSUFFICIENT" in raw:
        evaluation = "INSUFFICIENT_CONTEXT"
    else:
        evaluation = "AMBIGUOUS"

    return {"evaluation": evaluation}
