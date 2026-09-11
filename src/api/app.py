"""Tracer API - FastAPI wrapper around the pipeline (build) and chat engine.

Run from the project root (relative paths in config.py/session_store.py depend on it):
    uvicorn src.api.app:app --reload --port 8000

Then:
    GET  /health                       overall liveness
    GET  /pipeline/health               is the ingest pipeline ready to run (keys, /data present)
    POST /pipeline/build                kick off transcribe -> chunk -> embed in the background
    GET  /pipeline/status               poll progress of the last/current build
    GET  /chat/health                   is the chat engine ready (keys, index populated)
    POST /chat                          {"session_id": "...", "query": "..."} -> grounded answer
    GET  /episodes                      list episodes (number, title, duration)
    GET  /episodes/{episode_number}/audio   stream that episode's raw audio file

Interactive API docs at /docs once running.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Match this project's flat import convention (every other src/ module resolves via
# sys.path, not package-relative imports) rather than relying on `src` being importable
# as a dotted package - works the same regardless of how uvicorn is invoked.
API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import chat_routes  # noqa: E402
import episodes_routes  # noqa: E402
import pipeline_routes  # noqa: E402

app = FastAPI(title="Tracer API", version="1.0.0")

# Permissive CORS for a local React dev frontend. This is a local, single-user tool
# per the project's scope (no auth/multi-user infra) - tighten allow_origins if this
# is ever exposed beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_routes.router)
app.include_router(chat_routes.router)
app.include_router(episodes_routes.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "tracer-api"}
