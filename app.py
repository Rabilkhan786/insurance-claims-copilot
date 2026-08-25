"""FastAPI backend entry point for the Insurance Hybrid RAG application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from config import settings
from src.agent import (
    reset_session,
    run_agent,
    run_claim_review,
    stream_agent,
    submit_decision,
)
from src.tools.rag_tools import warmup as warmup_retriever
from src.utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    customer_id: str | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str | None
    session_id: str
    error: str | None = None


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1)


class ClaimRequest(BaseModel):
    """One claim as the employee typed it into the form."""

    customer_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    claim_amount: float = Field(gt=0)
    treatment: str = Field(min_length=1)
    diagnosis: str | None = None
    hospital: str | None = None
    treatment_date: str | None = None
    notes: str | None = None
    claim_id: str | None = None
    session_id: str | None = None


class ClaimResponse(BaseModel):
    session_id: str
    awaiting_review: bool
    recommendation: str | None = None
    eligibility: dict | None = None
    error: str | None = None


class DecisionRequest(BaseModel):
    """The employee's call on a recommendation that is waiting for review."""

    session_id: str = Field(min_length=1)
    decision: str = Field(pattern="^(approve|edit|reject)$")
    decided_by: str = Field(min_length=1)
    payable_amount: float | None = None
    edits: str | None = None
    reason: str | None = None


class DecisionResponse(BaseModel):
    session_id: str
    recorded: bool
    error: str | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Warm up the embedder and reranker once for each API process.

    The agent's tool modules (src/tools/*) each manage their own lazily
    built singletons -- this just triggers that build at startup instead
    of on the first request, so the first chat isn't slow.
    """
    try:
        warmup_retriever()
        application.state.agent_ready = True
    except Exception:
        logger.exception("startup_initialization_failed")
        application.state.agent_ready = False
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

app.state.agent_ready = False


def _require_agent_ready() -> None:
    if not app.state.agent_ready:
        raise HTTPException(
            503,
            "System is initializing. Please try again shortly.",
        )


@app.get("/")
async def root():
    """The frontend is streamlit_app.py; this API serves JSON only."""
    return {"message": "RAG AI Insurance Agent is running"}


@app.get("/health")
async def health():
    """Report whether the API process and RAG dependencies are available."""
    return {"status": "ok", "ready": app.state.agent_ready}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_session_id: Annotated[str | None, Header()] = None,
    x_customer_id: Annotated[str | None, Header()] = None,
):
    _require_agent_ready()
    session_id = x_session_id or request.session_id
    customer_id = x_customer_id or request.customer_id

    # stream=true returns plain text word by word instead of one JSON blob.
    if request.stream:
        return StreamingResponse(
            iterate_in_threadpool(
                stream_agent(request.question, session_id, customer_id)
            ),
            media_type="text/plain",
        )

    return await run_in_threadpool(
        run_agent, request.question, session_id, customer_id
    )


@app.post("/reset-memory")
async def reset_memory(request: ResetRequest):
    reset_session(request.session_id)
    return {"status": "reset", "session_id": request.session_id}


@app.post("/review-claim", response_model=ClaimResponse)
async def review_claim(request: ClaimRequest):
    """Analyse one claim and return a recommendation awaiting employee review.

    The graph pauses at the review step, so the response is a proposal, not a
    settled outcome. Finish it by POSTing to /submit-decision with the same
    session_id.
    """
    _require_agent_ready()
    claim = request.model_dump(
        exclude={"customer_id", "policy_id", "session_id"},
        exclude_none=True,
    )
    result = await run_in_threadpool(
        run_claim_review,
        claim,
        request.customer_id,
        request.policy_id,
        request.session_id,
    )
    return ClaimResponse(**result)


@app.post("/submit-decision", response_model=DecisionResponse)
async def submit_claim_decision(request: DecisionRequest):
    """Record the employee's decision and resume the paused review."""
    _require_agent_ready()
    result = await run_in_threadpool(
        submit_decision,
        request.session_id,
        request.decision,
        request.decided_by,
        request.payable_amount,
        request.edits,
        request.reason,
    )
    return DecisionResponse(**result)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=1,
    )
