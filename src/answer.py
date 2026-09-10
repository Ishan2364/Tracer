"""Phase 3: ties retrieval + refusal + generation together into one function.

Phase 5's evaluation harness calls answer_query() directly, per test case.

Phase 4 adds answer_conversational(), which layers intent parsing, episode-scoped/
broad/follow-up retrieval routing, and conversation state on top of the same
generate/refusal machinery - it does not replace answer_query().
"""

from __future__ import annotations

from langsmith import traceable

import config
import generate
import retrieval_router
from intent_parser import parse_intent
from retrieve import retrieve


@traceable(run_type="chain", name="answer_query")
def answer_query(query: str) -> dict:
    chunks = retrieve(query)
    retrieved_chunk_ids = [c["chunk_id"] for c in chunks]

    if not chunks or chunks[0]["similarity"] < config.SIMILARITY_THRESHOLD:
        return {
            "query": query,
            "answer": config.REFUSAL_MESSAGE,
            "citations": [],
            "refused": True,
            "retrieved_chunk_ids": retrieved_chunk_ids,
        }

    messages = generate.build_messages(query, chunks)
    answer_text = generate.call_llm(messages)

    manifest = generate.load_manifest()
    citations = []
    for c in chunks:
        info = manifest.get(c["episode_id"], {})
        citations.append({
            "episode_number": info.get("episode_number", c["episode_number"]),
            "episode_title": info.get("title", c["episode_id"]),
            "start": c["start"],
            "end": c["end"],
        })

    return {
        "query": query,
        "answer": answer_text,
        "citations": citations,
        "refused": False,
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }


def _build_citations(chunks: list[dict]) -> list[dict]:
    manifest = generate.load_manifest()
    citations = []
    for c in chunks:
        info = manifest.get(c["episode_id"], {})
        citations.append({
            "chunk_id": c["chunk_id"],
            "episode_number": info.get("episode_number", c.get("episode_number")),
            "episode_title": info.get("title", c["episode_id"]),
            "start": c["start"],
            "end": c["end"],
        })
    return citations


@traceable(run_type="chain", name="answer_conversational")
def answer_conversational(query: str, state) -> dict:
    """Phase 4 entrypoint: intent -> retrieval routing -> refusal -> generation,
    reading/writing the given ConversationState across turns."""
    intent = parse_intent(query, state.recent_history())
    query_type = intent["query_type"]

    routing = retrieval_router.route(intent, state)
    chunks = routing["chunks"]
    unknown_episodes = routing["unknown_episodes"]
    retrieval_mode = routing["retrieval_mode"]
    multi_episode = routing["multi_episode"]
    retrieved_chunk_ids = [c["chunk_id"] for c in chunks]

    base_result = {
        "query": query,
        "query_type": query_type,
        "episode_refs": intent.get("episode_refs", []),
        "retrieval_mode": retrieval_mode,
        "unknown_episodes": unknown_episodes,
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }

    # Every referenced episode was unknown and there's nothing else to fall back on.
    if not chunks and unknown_episodes:
        refs = ", ".join(str(n) for n in unknown_episodes)
        answer_text = f"Episode(s) {refs} don't exist in this catalogue."
        state.record(query, query_type, answer_text, chunks=[])
        return {**base_result, "answer": answer_text, "citations": [], "refused": False}

    # Catalogue-wide refusal check - only meaningful for broad/unscoped retrieval, where
    # "not covered" means the topic isn't in this catalogue at all. It does NOT apply to
    # follow-ups (grounding already validated on the turn that first retrieved it) or to
    # scoped_loop (the user named specific episodes, which we already know exist - a weak
    # similarity there just means *that phrasing* embedded thinly, not that the episode
    # lacks content; the system prompt's "say so if the excerpts don't answer it" instruction
    # handles per-episode irrelevance with more nuance than a hard binary refusal would).
    best_similarity = max((c.get("similarity", 1.0) for c in chunks), default=0.0)
    if retrieval_mode == "broad_rebalanced" and (not chunks or best_similarity < config.SIMILARITY_THRESHOLD):
        state.record(query, query_type, config.REFUSAL_MESSAGE, chunks=chunks)
        return {**base_result, "answer": config.REFUSAL_MESSAGE, "citations": [], "refused": True}

    if query_type == "follow_up":
        messages = generate.build_messages_followup(query, chunks, previous_answer=state.last_answer)
    elif multi_episode:
        messages = generate.build_messages_multi_episode(query, chunks, unknown_episodes)
    else:
        messages = generate.build_messages(query, chunks, unknown_episodes)

    answer_text = generate.call_llm(messages)
    citations = _build_citations(chunks)

    state.record(query, query_type, answer_text, chunks=chunks)

    return {**base_result, "answer": answer_text, "citations": citations, "refused": False}
