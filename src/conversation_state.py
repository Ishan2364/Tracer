"""Phase 4: in-memory conversation state, kept for the duration of one session."""

from __future__ import annotations

import config


class ConversationState:
    def __init__(self):
        self.history: list[dict] = []          # [{"query": ..., "answer": ...}, ...], most recent last
        self.last_query: str | None = None
        self.last_query_type: str | None = None
        self.last_retrieved_chunks: list[dict] = []
        self.last_answer: str | None = None

    def recent_history(self) -> list[dict]:
        return self.history[-config.CONVERSATION_HISTORY_TURNS:]

    def record(self, query: str, query_type: str, answer: str, chunks: list[dict] | None = None) -> None:
        """Record a completed turn. `chunks` is omitted only when a follow-up reused
        the existing grounding without expanding it - in that case last_retrieved_chunks
        is left untouched."""
        self.history.append({"query": query, "answer": answer})
        self.last_query = query
        self.last_query_type = query_type
        self.last_answer = answer
        if chunks is not None:
            self.last_retrieved_chunks = chunks
