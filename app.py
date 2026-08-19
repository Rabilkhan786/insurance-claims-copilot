"""FastAPI backend entry point for the Insurance Hybrid RAG application."""
from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config import settings
from src.embeddings import BGEEmbedder
from src.llm import GroqAnswerGenerator
from src.memory import ConversationMemory
from src.reranker import CrossEncoderReranker
from src.retrieval import HybridRetriever
from src.services import ChatService
from src.utils import configure_logging
from src.vectorstores import PineconeHybridStore
from src.voice import TranscriptionError, WhisperTranscriber

configure_logging()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str | None
    session_id: str
    error: str | None = None


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize shared RAG dependencies once for each API process."""
    try:
        embedder = BGEEmbedder()
        store = PineconeHybridStore(embedder)
        retriever = HybridRetriever(store, CrossEncoderReranker())
        application.state.chat_service = ChatService(
            retriever,
            application.state.memory,
            GroqAnswerGenerator(),
        )
    except Exception:
        logger.exception("startup_initialization_failed")
    yield


app = FastAPI(
    title="RAG AI Insurance Agent",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_directory = settings.root_dir / "static"
if static_directory.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=static_directory),
        name="static",
    )

app.state.chat_service = None
app.state.transcriber = None
app.state.memory = ConversationMemory()


def _chat_service() -> ChatService:
    if app.state.chat_service is None:
        raise HTTPException(
            503,
            "System is initializing. Please try again shortly.",
        )
    return app.state.chat_service


@app.get("/")
async def root():
    """Serve the retained frontend, when present."""
    index_page = static_directory / "index.html"
    if index_page.exists():
        return FileResponse(index_page)
    return {"message": "RAG AI Insurance Agent is running"}


@app.get("/health")
async def health():
    """Report whether the API process and RAG dependencies are available."""
    return {"status": "ok", "ready": app.state.chat_service is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_session_id: Annotated[str | None, Header()] = None,
):
    session_id = x_session_id or request.session_id
    return _chat_service().chat(request.question, session_id)


@app.post("/reset-memory")
async def reset_memory(request: ResetRequest):
    app.state.memory.reset(request.session_id)
    return {"status": "reset", "session_id": request.session_id}


@app.post("/voice-chat", response_model=ChatResponse)
async def voice_chat(
    audio: Annotated[UploadFile, File(...)],
    session_id: str | None = None,
    x_session_id: Annotated[str | None, Header()] = None,
):
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as target:
        target.write(await audio.read())
        audio_path = Path(target.name)

    try:
        if app.state.transcriber is None:
            app.state.transcriber = WhisperTranscriber()

        try:
            question = await run_in_threadpool(
                app.state.transcriber.transcribe,
                audio_path,
            )
        except TranscriptionError as error:
            raise HTTPException(422, str(error)) from error

        resolved_session_id = x_session_id or session_id
        return _chat_service().chat(question, resolved_session_id)
    finally:
        audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=1,
    )
