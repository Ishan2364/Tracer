"""Phase 3: ties retrieval + refusal + generation together into one function.

Phase 5's evaluation harness calls answer_query() directly, per test case.

Phase 4 adds answer_conversational(), which layers intent parsing, episode-scoped/
broad/follow-up retrieval routing, and conversation state on top of the same
generate/refusal machinery - it does not replace answer_query().
"""

from __future__ import annotations

import json

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

    messages, chunks_sent = generate.build_messages(query, chunks)
    answer_text = generate.call_llm(messages)

    manifest = generate.load_manifest()
    citations = []
    for c in chunks_sent:
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


def _run_chitchat(query: str, state) -> dict:
    """Gate 2: even if the intent classifier (Gate 1) mis-routes a real content
    question as chitchat, this call is explicitly forbidden from answering it from
    its own knowledge - it can only reply warmly or admit the message actually needs
    a search, via the needs_retrieval flag. Defaults to needs_retrieval=True (i.e.
    falls back to a real search) if the structured output can't even be parsed -
    the safe failure mode is "search anyway", never "trust an unparseable reply"."""
    messages = generate.build_messages_chitchat(query, state.recent_history())
    raw = generate.call_llm(messages, model=config.INTENT_MODEL, response_format={"type": "json_object"})
    try:
        data = json.loads(raw)
        return {"reply": data["reply"], "needs_retrieval": bool(data.get("needs_retrieval", False))}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"reply": "", "needs_retrieval": True}


def _handle_recommendation_no_topic(query: str, state) -> dict:
    """A recommendation request with no identifiable topic ('recommend me an episode',
    'what should I watch next') has nothing to run similarity search against - embedding
    an empty topic string produces a near-arbitrary vector, which then fails the broad-
    retrieval refusal-gate's similarity check and gets reported as 'not covered'. That's
    wrong: there's no topic to be uncovered, the user simply hasn't said what they're
    interested in yet - a completely different situation from a real off-topic query.

    Handled the same way catalogue is - deterministically from the manifest, no
    retrieval, no generation call - and asks them to name an interest rather than
    silently refusing, or guessing a "best" episode with nothing to ground that claim in
    (there's no difficulty/quality metadata to base such a pick on - inventing one would
    be exactly the kind of ungrounded claim this product is built to avoid elsewhere)."""
    with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest = sorted(manifest, key=lambda e: e["episode_number"])

    lines = [
        "Happy to point you to something - what are you interested in? For example: "
        "relativity, black holes, DNA, information theory, neural networks, "
        "computability, or evolution.",
        "",
        f"Or browse the full collection ({len(manifest)} episodes):",
    ]
    for e in manifest:
        title = generate.derive_title(e["source_file"])
        lines.append(f"- Episode {e['episode_number']}: {title}")
    answer_text = "\n".join(lines)

    state.record(query, "recommendation", answer_text, chunks=[])
    return {
        "query": query,
        "query_type": "recommendation",
        "episode_refs": [],
        "retrieval_mode": "recommendation_no_topic",
        "unknown_episodes": [],
        "retrieved_chunk_ids": [],
        "answer": answer_text,
        "citations": [],
        "refused": False,
    }


def _handle_catalogue(query: str, state) -> dict:
    """Inventory questions ('what episodes do you have') can never be answered
    correctly by semantic retrieval - top-k similarity search returns whichever
    chunks happen to score highest, not every episode, so presenting that as "the
    collection" is a completeness claim retrieval structurally cannot back up (this
    is exactly the bug found in practice: three different partial, mutually-
    inconsistent episode lists across three near-identical questions in one
    session). The manifest is the actual ground truth - read it directly, no
    retrieval, no generation call, so the list is always complete and correct."""
    with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest = sorted(manifest, key=lambda e: e["episode_number"])

    lines = [f"The collection has {len(manifest)} episode(s):"]
    for e in manifest:
        title = generate.derive_title(e["source_file"])
        duration = generate.format_timestamp(e["duration_seconds"]) if e.get("duration_seconds") else "?"
        lines.append(f"- Episode {e['episode_number']}: {title} ({duration})")
    answer_text = "\n".join(lines)

    state.record(query, "catalogue", answer_text, chunks=[])
    return {
        "query": query,
        "query_type": "catalogue",
        "episode_refs": [],
        "retrieval_mode": "catalogue",
        "unknown_episodes": [],
        "retrieved_chunk_ids": [],
        "answer": answer_text,
        "citations": [],
        "refused": False,
    }


@traceable(run_type="chain", name="answer_conversational")
def answer_conversational(query: str, state) -> dict:
    """Phase 4 entrypoint: intent -> retrieval routing -> refusal -> generation,
    reading/writing the given ConversationState across turns."""
    intent = parse_intent(query, state.recent_history())
    query_type = intent["query_type"]

    if query_type == "catalogue":
        return _handle_catalogue(query, state)

    if query_type == "recommendation" and not intent.get("topic", "").strip():
        return _handle_recommendation_no_topic(query, state)

    if query_type == "chitchat":
        chitchat = _run_chitchat(query, state)
        if not chitchat["needs_retrieval"]:
            answer_text = chitchat["reply"]
            state.record(query, "chitchat", answer_text, chunks=[])
            return {
                "query": query,
                "query_type": "chitchat",
                "episode_refs": [],
                "retrieval_mode": "chitchat",
                "unknown_episodes": [],
                "retrieved_chunk_ids": [],
                "answer": answer_text,
                "citations": [],
                "refused": False,
            }
        # Gate 2 caught a Gate 1 misclassification - re-run classification on the raw
        # query now that we know it's a real content question, so episode_refs/topic/
        # time_range get properly extracted instead of falling back to a bare unscoped
        # search. Gate 1's own first-pass fields aren't trustworthy here even though
        # parse_intent always returns them - they "weren't extracted carefully since it
        # thought this wasn't a content question at all" (see below), same reason a
        # generic fallback was used originally.
        #
        # Guarded against looping: classification isn't perfectly deterministic
        # (observed directly - repeated identical calls to the same query have
        # returned different results), so if this second pass also says chitchat,
        # don't call _run_chitchat again - fall back to the safe generic search
        # instead, exactly as before.
        reclassified = parse_intent(query, state.recent_history())
        if reclassified["query_type"] != "chitchat":
            intent = reclassified
            query_type = reclassified["query_type"]
        else:
            intent = {"query_type": "general", "topic": query, "episode_refs": [], "time_range": None}
            query_type = "general"

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

    # Nothing was retrieved even though every referenced episode is real - e.g. a
    # time-range query whose window genuinely has no matching content, or a follow_up
    # with no prior turn to build on. Say so deterministically rather than calling the
    # generation model on empty grounding and hoping it declines on its own.
    if not chunks and not unknown_episodes:
        if query_type == "follow_up":
            answer_text = "There's no earlier answer to build on yet - try asking a full question first."
        else:
            answer_text = "Nothing in the specified part of that episode matches this question."
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
        # For `recommendation` specifically, weak similarity across the board almost
        # always means "no real topic was given" (e.g. "what should I watch next"),
        # not "this topic isn't covered" - a recommendation request has no off-topic
        # canary equivalent the way a content question does. The intent classifier's
        # `topic` extraction for a genuinely topic-less request isn't perfectly stable
        # across calls even at temperature=0 (confirmed: 5 identical calls to "what
        # episode should I watch next?" returned topic="" three times and "next
        # episode" twice) - checking the actual retrieval outcome here catches the
        # vacuous-topic case regardless of which non-empty-but-meaningless string the
        # classifier happened to extract, instead of only catching the literal "" case.
        if query_type == "recommendation":
            return _handle_recommendation_no_topic(query, state)
        state.record(query, query_type, config.REFUSAL_MESSAGE, chunks=chunks)
        return {**base_result, "answer": config.REFUSAL_MESSAGE, "citations": [], "refused": True}

    if query_type == "follow_up":
        messages, chunks_sent = generate.build_messages_followup(query, chunks, previous_answer=state.last_answer)
    elif query_type == "recommendation":
        messages, chunks_sent = generate.build_messages_recommendation(query, chunks, unknown_episodes)
    elif query_type == "broad_comparison" or multi_episode:
        # broad_comparison always gets the per-episode comparative framing, even if this
        # particular query's retrieval happened to land on just one episode - the user
        # explicitly asked for a cross-episode comparison, so "only episode 6 covers this"
        # is itself the comparison answer, not a reason to fall back to the plain prompt.
        messages, chunks_sent = generate.build_messages_multi_episode(query, chunks, unknown_episodes)
    else:
        messages, chunks_sent = generate.build_messages(query, chunks, unknown_episodes)

    answer_text = generate.call_llm(messages)
    # First narrow chunks_sent (everything actually in the prompt - see below) down to
    # used_chunks (only what the answer's own inline citations actually point to), then
    # build the displayed citations from that narrower set. Two separate corrections
    # stacked here: chunks_sent fixes "retrieved but never sent" (budget trimming can
    # drop chunks before they reach the model), filter_citations_to_used fixes "sent but
    # never actually cited" (the model was shown it but didn't end up using it). Neither
    # step ever trusts a new model claim - both only narrow down which subset of
    # already-verified real chunk metadata gets displayed.
    used_chunks = generate.filter_citations_to_used(answer_text, chunks_sent)
    citations = _build_citations(used_chunks)

    state.record(query, query_type, answer_text, chunks=chunks)

    return {**base_result, "answer": answer_text, "citations": citations, "refused": False}
