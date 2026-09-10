"""Session-keyed persistence for ConversationState: session_id -> state, as one
JSON file per session on disk.

This is the appropriately-scoped version for this project - the brief's non-goals
explicitly exclude production/multi-user infrastructure, so a disk-backed dict is
the right tool, not Redis. The interface (load_session/save_session) is what would
change if this ever became a real multi-worker backend - swap the file read/write
below for a Redis get/setex, ConversationState.to_dict()/from_dict() stay the same.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from conversation_state import ConversationState

SESSIONS_DIR = Path("sessions")
_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]+")


def _session_path(session_id: str) -> Path:
    safe_id = _SAFE_ID.sub("", session_id)
    if not safe_id:
        raise ValueError(f"invalid session_id: {session_id!r}")
    return SESSIONS_DIR / f"{safe_id}.json"


def load_session(session_id: str) -> ConversationState:
    path = _session_path(session_id)
    if not path.exists():
        return ConversationState()
    with open(path, "r", encoding="utf-8") as f:
        return ConversationState.from_dict(json.load(f))


def save_session(session_id: str, state: ConversationState) -> None:
    path = _session_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
