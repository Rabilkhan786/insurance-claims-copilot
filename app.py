"""FastAPI backend for the Claims Copilot."""
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
    ClaimRecommendation,
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
    """Claim details submitted by an employee."""

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
    """Claim recommendation waiting for employee review."""

    session_id: str
    awaiting_review: bool
    recommendation: ClaimRecommendation | None = None
    error: str | None = None


class DecisionRequest(BaseModel):
    """Employee decision for a paused claim review."""

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
    """Warm up retrieval models when the API starts."""
    try:
        warmup_retriever()
        application.state.agent_ready = True
    except Exception:
        logger.exception("startup_initialization_failed")
        application.state.agent_ready = False
    yield


app = FastAPI(
    title="AI Health Insurance Claims Copilot",
    version="1.0.0",
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
    return {"message": "AI Health Insurance Claims Copilot is running"}


@app.get("/health")
async def health():
    """Return API and retrieval readiness."""
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

    if request.stream:
        return StreamingResponse(
            iterate_in_threadpool(
                stream_agent(request.question, session_id, customer_id)
            ),
            media_type="text/plain",
        )

    return await run_in_threadpool(
        run_agent,
        request.question,
        session_id,
        customer_id,
    )


@app.post("/reset-memory")
async def reset_memory(request: ResetRequest):
    reset_session(request.session_id)
    return {"status": "reset", "session_id": request.session_id}


@app.post("/review-claim", response_model=ClaimResponse)
async def review_claim(request: ClaimRequest):
    """Analyse a claim and pause for employee review."""
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
    """Save the employee decision and resume the paused review."""
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
