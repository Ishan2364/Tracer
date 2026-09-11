"""Phase 4: routes a parsed intent to the right retrieval strategy (the §6 decision table)."""

from __future__ import annotations

import json
from pathlib import Path

import config
from episode_resolver import resolve_episodes
from retrieve import retrieve, retrieve_scoped, retrieve_scoped_timerange

_episode_chunk_cache: dict[str, list[dict]] = {}


def _load_episode_chunks(episode_id: str) -> list[dict]:
    if episode_id not in _episode_chunk_cache:
        path = Path(config.CHUNKS_DIR) / f"{episode_id}.json"
        with open(path, "r", encoding="utf-8") as f:
            _episode_chunk_cache[episode_id] = json.load(f)
    return _episode_chunk_cache[episode_id]


def rebalance(chunks: list[dict], cap: int = config.PER_EPISODE_CAP) -> list[dict]:
    """Keep at most `cap` chunks per episode_id, preserving overall similarity order,
    so one dense/long episode can't crowd out every slot."""
    counts: dict[str, int] = {}
    kept = []
    for c in chunks:
        eid = c["episode_id"]
        if counts.get(eid, 0) < cap:
            kept.append(c)
            counts[eid] = counts.get(eid, 0) + 1
    return kept


def expand_context(chunks: list[dict], window: int = config.FOLLOWUP_CONTEXT_WINDOW) -> list[dict]:
    """For each chunk, pull in the `window` chunks immediately before/after it (by
    position within its episode's chunk file) - no new Chroma call, just local reads.

    Returns originally-retrieved chunks first (chronologically sorted among
    themselves), followed by the expanded neighbor-only chunks (also chronological).
    This ordering matters downstream: generate.py's budget trim just keeps chunks in
    the order it's given until the token budget runs out, so putting the actually-
    retrieved chunks first guarantees they survive trimming - a follow-up is almost
    always about something in one of those, not an incidental neighbor - instead of
    losing whichever chunk happens to sort latest by raw timestamp."""
    original_ids = {c["chunk_id"] for c in chunks}
    expanded: dict[str, dict] = {}
    neighbor_ids: set[str] = set()

    for c in chunks:
        expanded[c["chunk_id"]] = c
        episode_chunks = _load_episode_chunks(c["episode_id"])
        idx = next((i for i, ec in enumerate(episode_chunks) if ec["chunk_id"] == c["chunk_id"]), None)
        if idx is None:
            continue
        for offset in range(-window, window + 1):
            j = idx + offset
            if offset == 0 or not (0 <= j < len(episode_chunks)):
                continue
            neighbor = episode_chunks[j]
            if neighbor["chunk_id"] not in expanded:
                expanded[neighbor["chunk_id"]] = neighbor
                neighbor_ids.add(neighbor["chunk_id"])

    def _chrono(c):
        return (c["episode_id"], c["start"])

    original = sorted((c for cid, c in expanded.items() if cid in original_ids), key=_chrono)
    neighbors = sorted((c for cid, c in expanded.items() if cid in neighbor_ids), key=_chrono)
    return original + neighbors


def route(intent: dict, state) -> dict:
    """Returns {"chunks": [...], "unknown_episodes": [...], "retrieval_mode": "...", "multi_episode": bool}."""
    query_type = intent["query_type"]
    topic = intent.get("topic") or ""
    episode_refs = intent.get("episode_refs") or []

    if query_type == "follow_up":
        chunks = expand_context(state.last_retrieved_chunks) if state.last_retrieved_chunks else []
        return {
            "chunks": chunks,
            "unknown_episodes": [],
            "retrieval_mode": "follow_up",
            "multi_episode": len({c["episode_id"] for c in chunks}) > 1,
        }

    is_named = query_type in ("single_episode", "named_comparison")
    if is_named and episode_refs and len(episode_refs) <= config.NAMED_COMPARISON_MAX:
        resolved, unknown = resolve_episodes(episode_refs)
        time_range = intent.get("time_range")
        chunks = []
        for ep in resolved:
            if time_range:
                chunks.extend(retrieve_scoped_timerange(
                    topic or " ", ep["episode_id"], time_range[0], time_range[1], n_results=config.N_RESULTS
                ))
            else:
                chunks.extend(retrieve_scoped(topic or " ", ep["episode_id"], n_results=config.N_RESULTS))
        return {
            "chunks": chunks,
            "unknown_episodes": unknown,
            "retrieval_mode": "scoped_loop",
            "multi_episode": len(resolved) > 1,
        }

    # broad_comparison / general / recommendation / named_comparison with >6 refs /
    # single_episode or named_comparison that named no resolvable episode.
    raw = retrieve(topic, n_results=config.BROAD_N_RESULTS)
    chunks = rebalance(raw, cap=config.PER_EPISODE_CAP)
    return {
        "chunks": chunks,
        "unknown_episodes": [],
        "retrieval_mode": "broad_rebalanced",
        "multi_episode": len({c["episode_id"] for c in chunks}) > 1,
    }
