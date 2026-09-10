"""Endpoints for the conversational engine (Phases 3-4) - thin HTTP wrapper
around answer_conversational(), with the same session_id-keyed persistence
already used by `chat.py --session-id`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parent.parent  # src/api/ -> src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from answer import answer_conversational  # noqa: E402
from retrieve import get_collection  # noqa: E402
from session_store import load_session, save_session  # noqa: E402

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    query: str


class Citation(BaseModel):
    chunk_id: str | None = None
    episode_number: int | None = None
    episode_title: str | None = None
    start: float
    end: float


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    query_type: str
    retrieval_mode: str
    refused: bool
    unknown_episodes: list[int]


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id must not be empty")

    state = load_session(req.session_id)
    result = answer_conversational(req.query, state)
    save_session(req.session_id, state)

    return ChatResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        query_type=result["query_type"],
        retrieval_mode=result["retrieval_mode"],
        refused=result["refused"],
        unknown_episodes=result["unknown_episodes"],
    )


@router.get("/health")
def chat_health():
    groq_key_set = bool(os.environ.get("GROQ_API_KEY"))
    langsmith_tracing = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"

    indexed_chunk_count = None
    index_error = None
    try:
        indexed_chunk_count = get_collection().count()
    except Exception as exc:
        index_error = str(exc)

    ok = groq_key_set and indexed_chunk_count is not None and indexed_chunk_count > 0
    return {
        "status": "ok" if ok else "degraded",
        "groq_api_key_set": groq_key_set,
        "langsmith_tracing_enabled": langsmith_tracing,
        "indexed_chunk_count": indexed_chunk_count,
        "index_error": index_error,
        "similarity_threshold": config.SIMILARITY_THRESHOLD,
        "generation_model": config.GENERATION_MODEL,
        "intent_model": config.INTENT_MODEL,
    }
