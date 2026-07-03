# Phase 4 — Multi-turn Conversation Memory

## Background

Every `/chat` call today initialises a fresh `AgentState` with no history. Users cannot ask follow-up questions like "what else did he say about that?" because the agent has no memory of the previous exchange. This makes the assistant feel like a search engine, not a conversation partner.

LangGraph's `MemorySaver` checkpointer stores the full message history keyed by a `thread_id`. Passing a stable `session_id` from the frontend ties all turns of a conversation together without any manual history management.

**Scope:** `backend/agents/state.py`, `backend/agents/graph.py`, `backend/agents/synthesis.py`, `backend/routers/chat.py`, `backend/static/index.html`.

---

## Tasks

### 1. Extend `AgentState` in `backend/agents/state.py`
- [ ] Add `from typing import Annotated` and `from langgraph.graph.message import add_messages`
- [ ] Add a `messages: Annotated[list, add_messages]` field to `AgentState`
  - `add_messages` is a LangGraph reducer that appends new messages instead of overwriting the list

### 2. Wire `MemorySaver` into `backend/agents/graph.py`
- [ ] Import `from langgraph.checkpoint.memory import MemorySaver`
- [ ] Instantiate `_checkpointer = MemorySaver()` as a module-level singleton
- [ ] In `build_graph()`, compile with `g.compile(checkpointer=_checkpointer)` instead of `g.compile()`

### 3. Update `backend/agents/synthesis.py`
- [ ] Before calling Ollama for synthesis, prepend `state["messages"]` as prior conversation turns in the messages list sent to the model
- [ ] Append the current Q&A pair (`HumanMessage(query)` + `AIMessage(answer)`) to `state["messages"]` after synthesis so the next turn can see it

### 4. Update `backend/routers/chat.py`
- [ ] Add `session_id: str | None = None` to `ChatRequest`
- [ ] In the `POST /chat` handler, build a LangGraph `config`:
  ```python
  config = {"configurable": {"thread_id": req.session_id or "default"}}
  ```
- [ ] Pass `config` to `graph.ainvoke(init_state, config=config)`
- [ ] Add `session_id: str | None = None` to `_sse_generator` signature and thread the same config through `GET /chat/stream`

### 5. Update `backend/static/index.html`
- [ ] On page load, generate a session ID: `const sessionId = crypto.randomUUID()`
- [ ] Include `session_id: sessionId` in every `POST /chat` request body
- [ ] Include `&session_id=<sessionId>` as a query param on `GET /chat/stream` requests

---

## Acceptance Criteria

- [ ] `AgentState` contains a `messages` field using the `add_messages` reducer
- [ ] `build_graph()` compiles with `MemorySaver` as the checkpointer
- [ ] `ChatRequest` accepts an optional `session_id` field
- [ ] Sending two related questions with the same `session_id` results in the second answer referencing context from the first (verified manually)
- [ ] Sending a question with no `session_id` still works (falls back to `"default"` thread)
- [ ] `GET /chat/stream` (SSE) respects the same session threading
- [ ] `POST /chat` response contract is unchanged (`query`, `answer`, `evaluation`, `sources`)
- [ ] The frontend generates and reuses a `session_id` across the full browser session

---

## Notes

- `MemorySaver` stores history in-process memory — it resets on server restart. This is intentional for now; swap in `SqliteSaver` or a Redis-backed checkpointer in a future issue if persistence across restarts is needed.
- No new packages are required — `MemorySaver` ships with `langgraph`.
- Keep `session_id` optional so existing clients (curl, tests) that omit it continue to work.
