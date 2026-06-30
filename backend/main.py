from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routers import chat, podcast, upload

app = FastAPI(
    title="Multi-Modal Knowledge Agent",
    version="0.1.0",
    description="Audio ingestion, transcription, and RAG-ready vector storage.",
)

app.include_router(upload.router)
app.include_router(podcast.router)
app.include_router(chat.router)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
