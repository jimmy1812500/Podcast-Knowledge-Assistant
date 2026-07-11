from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query: str  # original user question (never mutated)
    search_query: str  # may be rephrased during query expansion
    context: list[dict]  # retrieved chunks from ChromaDB
    evaluation: str  # CONFIDENT | INSUFFICIENT_CONTEXT | AMBIGUOUS
    answer: str
    iteration: int  # retrieval attempt count (max = MAX_ITERATIONS in graph)
    messages: Annotated[list, add_messages]
    user_id: str  # scopes ChromaDB retrieval to the requesting user
