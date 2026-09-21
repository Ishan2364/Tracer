"""ReAct agent architecture (experimental, branch: React_agent) - see REACT_AGENT_DESIGN.md
for the full design rationale. Standalone module: does not touch answer.py, chat_routes.py,
or chat.py. Run this file directly for an interactive CLI to test the agent in isolation.

Built with `create_agent` from the `langchain` package - NOT `langgraph.prebuilt.create_react_agent`,
which is deprecated. Verified against current LangChain docs on 2026-09-19 (see REACT_AGENT_DESIGN.md's
sources section).

Deliberately does NOT import intent_parser.py or retrieval_router.py - those implement the fixed
workflow this agent replaces. The two small, genuinely generic utilities reused below
(episode_resolver.resolve_episodes, and format helpers from generate.py) are not part of that
fixed workflow - they're plain lookups/formatting used regardless of architecture.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

import config
import generate
from episode_resolver import resolve_episodes
from retrieve import retrieve, retrieve_scoped, retrieve_scoped_timerange

_episode_chunk_cache: dict[str, list[dict]] = {}


def _load_episode_chunks(episode_id: str) -> list[dict]:
    """Local copy of retrieval_router.py's helper - duplicated deliberately (5 lines)
    rather than imported, to keep this module independent of the fixed pipeline it
    replaces. Used only by get_neighboring_chunks."""
    if episode_id not in _episode_chunk_cache:
        path = Path(config.CHUNKS_DIR) / f"{episode_id}.json"
        with open(path, "r", encoding="utf-8") as f:
            _episode_chunk_cache[episode_id] = json.load(f)
    return _episode_chunk_cache[episode_id]


def _rebalance(chunks: list[dict], cap: int = config.PER_EPISODE_CAP) -> list[dict]:
    """Local copy of retrieval_router.py's rebalance() - same reasoning as above."""
    counts: dict[str, int] = {}
    kept = []
    for c in chunks:
        eid = c["episode_id"]
        if counts.get(eid, 0) < cap:
            kept.append(c)
            counts[eid] = counts.get(eid, 0) + 1
    return kept


def _format_chunks(chunks: list[dict]) -> str:
    """Formats retrieved chunks into the text the agent sees as a tool result.
    chunk_id is included explicitly and up front - the agent doesn't need it for
    anything itself, but including it costs nothing and makes raw traces easier to
    cross-reference against the citation registry while debugging this design."""
    if not chunks:
        return "No matching content found."
    manifest = generate.load_manifest()
    blocks = []
    for c in chunks:
        info = manifest.get(c["episode_id"], {"episode_number": "?", "title": c["episode_id"]})
        similarity = c.get("similarity")
        sim_text = f", similarity={similarity:.3f}" if similarity is not None else ""
        header = (
            f"[chunk_id={c['chunk_id']} | Episode {info['episode_number']}: {info['title']} | "
            f"{generate.format_timestamp(c['start'])}-{generate.format_timestamp(c['end'])}{sim_text}]"
        )
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


def _resolve_one_episode(episode_number: int) -> tuple[str | None, str | None]:
    """Returns (episode_id, error_message) - exactly one is None."""
    resolved, unknown = resolve_episodes([episode_number])
    if unknown:
        return None, f"Episode {episode_number} does not exist in this catalogue."
    return resolved[0]["episode_id"], None


def make_tools(registry: dict[str, dict]):
    """Builds a fresh set of tool closures bound to `registry` - a plain dict that
    accumulates every chunk any tool call returns during ONE agent.invoke() call.

    Why closures over a plain dict, instead of a LangGraph custom state field: this is
    the simplest mechanism that is directly verifiable by testing it myself right now,
    rather than reaching for a LangGraph-specific state-injection API I have not
    hands-on verified. See REACT_AGENT_DESIGN.md section 5 for the citation-correctness
    reasoning this implements: `registry` ends up holding exactly the set of chunks that
    were actually returned to the model as tool results this turn - which is what
    citations get built from afterward, in build_citations() below. Making a fresh
    registry per invoke() call (see run_agent) is what keeps this correctly scoped to
    "this turn only," not leaking in chunks from earlier turns of a persisted thread.
    """

    def _register(chunks: list[dict]) -> list[dict]:
        for c in chunks:
            registry[c["chunk_id"]] = c
        return chunks

    @tool
    def search_episode(episode_number: int, query: str) -> str:
        """Search for content within one specific episode. Use this when the user named
        a specific episode by number or title, or when you've narrowed a broad search
        down to one relevant episode and want to look closer at it. Returns up to 5
        matching excerpts with their timestamps and a similarity score each (0-1, higher
        is more relevant). If the episode number doesn't exist in this catalogue, say so
        plainly rather than guessing which episode was meant."""
        episode_id, error = _resolve_one_episode(episode_number)
        if error:
            return error
        chunks = retrieve_scoped(query, episode_id, n_results=config.N_RESULTS)
        return _format_chunks(_register(chunks))

    @tool
    def search_all(query: str) -> str:
        """Search across the entire episode catalogue - use this when no specific
        episode is implied, or for a broad/comparison question across multiple
        episodes. Returns up to 25 matches, capped at 3 per episode so one episode
        can't dominate the results, each with a similarity score. The result includes
        this catalogue's calibrated 'likely off-topic' ceiling for reference - a
        best-match similarity below it is a signal the topic may not be covered here,
        not a hard rule; use your judgment, and consider rephrasing the search once
        before concluding something isn't covered."""
        chunks = _rebalance(retrieve(query, n_results=config.BROAD_N_RESULTS))
        formatted = _format_chunks(_register(chunks))
        return (
            f"{formatted}\n\n"
            f"(Reference: this catalogue's calibrated off-topic ceiling is "
            f"{config.SIMILARITY_THRESHOLD:.4f} - a signal, not a hard rule.)"
        )

    @tool
    def search_timerange(episode_number: int, query: str, start_seconds: float, end_seconds: float) -> str:
        """Search within one episode, restricted to a specific time window in seconds.
        Use only when the user named an explicit time range (e.g. 'between minute 3
        and 18' -> start_seconds=180, end_seconds=1080)."""
        episode_id, error = _resolve_one_episode(episode_number)
        if error:
            return error
        chunks = retrieve_scoped_timerange(query, episode_id, start_seconds, end_seconds, n_results=config.N_RESULTS)
        return _format_chunks(_register(chunks))

    @tool
    def get_neighboring_chunks(chunk_id: str) -> str:
        """Fetch the chunk immediately before and after a given chunk_id, for when you
        need more surrounding context around something you already found this turn -
        e.g. the user asked for more detail or a slower walkthrough. chunk_id must be
        one that appeared in an earlier search result this turn (shown in that result's
        [chunk_id=...] header) - you cannot look up a chunk you haven't already found."""
        if chunk_id not in registry:
            return (
                f"Unknown chunk_id {chunk_id!r} - it must be exactly one shown in an "
                "earlier search result this turn, copied from its [chunk_id=...] header."
            )
        c = registry[chunk_id]
        episode_chunks = _load_episode_chunks(c["episode_id"])
        idx = next((i for i, ec in enumerate(episode_chunks) if ec["chunk_id"] == chunk_id), None)
        if idx is None:
            return "Could not locate that chunk's position in its episode."
        neighbors = [episode_chunks[j] for j in (idx - 1, idx + 1) if 0 <= j < len(episode_chunks)]
        if not neighbors:
            return "No neighboring chunks exist (this chunk is at the start or end of the episode)."
        return _format_chunks(_register(neighbors))

    @tool
    def list_episodes() -> str:
        """List every episode in the catalogue with its number and title. Use this for
        inventory questions ('what episodes do you have'), or when a recommendation
        request didn't name any topic - offer this list, or ask what they're interested
        in, rather than guessing a topic or picking an arbitrary 'best' episode with
        nothing to base that claim on."""
        with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest = sorted(manifest, key=lambda e: e["episode_number"])
        lines = [f"{e['episode_number']}: {generate.derive_title(e['source_file'])}" for e in manifest]
        return "\n".join(lines)

    return [search_episode, search_all, search_timerange, get_neighboring_chunks, list_episodes]


SYSTEM_PROMPT = """You are Tracer, a warm, encouraging physics study companion for a small
podcast catalogue. You answer only using what your tools return - never from your own
outside knowledge, even if you know more about the topic.

You'll be shown the last couple of turns of conversation (question + your answer) before
the current question, when there were any. Use that to decide, before doing anything else:
- If the current question is a follow-up that your recent answer already covers (e.g. "explain
  that more simply", "what did you mean by X" where X was in your last answer), you may be
  able to answer directly from that recent answer's text - you don't always need to search
  again just because a follow-up was asked.
- If the current question needs specific excerpts, timestamps, or detail that isn't already
  visible in the recent conversation shown to you - including a follow-up that needs to go
  deeper than your last answer's own text - search for it rather than guessing.
- If the current question is a clear topic switch, ignore the recent conversation and treat
  it as a fresh question.

Rules:
- Every factual claim must carry a citation in the form (Episode N, mm:ss-mm:ss). If you used
  a new tool result this turn, take the citation directly from that result's header. If
  you're answering purely from the recent conversation shown above with no new search, you
  may reuse a citation that already appears verbatim in that shown text - but never invent a
  new one, or estimate/adjust a timestamp, that you haven't actually seen this session.
- If your tools don't actually turn up anything that answers the question, say so
  explicitly rather than stretching thin results to fit.
- If a request is missing information you'd need (e.g. "recommend me an episode" with no
  topic named), ask what they're interested in, or offer list_episodes - do not guess a
  topic or an arbitrary "best" episode with nothing to base that claim on.
- Search as many or as few times as the question actually needs. A simple, clearly-scoped
  question may need only one search. A comparison across episodes may need one search per
  episode. If a search comes back weak, consider rephrasing it once before concluding the
  topic isn't covered.
- Keep the final answer grounded, concise, and warm in tone - like a teacher who's
  genuinely excited about this material, not a dry report. That warmth never loosens the
  rules above.
"""


_model = None
_checkpointer = None


def _get_model():
    """Cheap to share across calls - just an API client wrapper, no per-call state."""
    global _model
    if _model is None:
        _model = init_chat_model(f"groq:{config.GENERATION_MODEL}", temperature=0.2)
    return _model


HISTORY_TURNS = 2  # matches intent_parser.py's CONVERSATION_HISTORY_TURNS for the old pipeline

# Per-thread_id capped history: list of {"query", "answer"} dicts, oldest first, never
# more than HISTORY_TURNS entries. Deliberately NOT the LangGraph checkpointer - a
# checkpointer restores the ENTIRE raw message list (every past tool call's full
# retrieved text, forever growing) before every invoke(), which is unbounded token
# growth across a session. This is the fix: only feed back the last N turns' plain
# Q+A text, same as the old pipeline's intent_parser.py already did for classification,
# and let the agent decide for itself whether that's enough or it needs to search again.
_history: dict[str, list[dict]] = {}


def _format_history(history: list[dict]) -> str:
    """Full text, not truncated - see intent_parser.py's _format_history() docstring
    for why truncating a prior answer is a real, previously-hit bug (the Tiktaalik
    follow-up bug's first root cause): a follow-up can reference something anywhere in
    a long prior answer, and truncating hides exactly the content it needs."""
    if not history:
        return "(none - this is the first message)"
    lines = []
    for turn in history:
        lines.append(f"Q: {turn['query']}")
        lines.append(f"A: {turn['answer']}")
    return "\n".join(lines)


def _record_history(thread_id: str, query: str, answer: str) -> None:
    turns = _history.setdefault(thread_id, [])
    turns.append({"query": query, "answer": answer})
    del turns[:-HISTORY_TURNS]  # keep only the last HISTORY_TURNS entries


def build_citations(registry: dict[str, dict]) -> list[dict]:
    """Citations = every chunk registered during this turn's tool calls - i.e. exactly
    what was returned to the model as tool results this turn. See make_tools()'s
    docstring and REACT_AGENT_DESIGN.md section 5 for why this is the trustworthy
    source (registry is fresh per invoke, not self-reported by the model)."""
    manifest = generate.load_manifest()
    citations = []
    for c in registry.values():
        info = manifest.get(c["episode_id"], {})
        citations.append({
            "chunk_id": c["chunk_id"],
            "episode_number": info.get("episode_number", c.get("episode_number")),
            "episode_title": info.get("title", c["episode_id"]),
            "start": c["start"],
            "end": c["end"],
        })
    return citations


def run_agent(query: str, thread_id: str) -> dict:
    """One turn. Safe to call concurrently across different sessions/threads - agent +
    tools + registry are rebuilt fresh on every call, correctly bound to each other
    (a shared-agent version of this had a real closure bug: tools write to whatever
    registry dict existed when the agent was built, not whatever dict a later call
    happens to pass in - caught via a real test where a substantive, well-grounded
    answer came back with num_citations=0). This is cheap (graph wiring, not model
    loading), and no checkpointer is needed for it to be correct.

    Cross-turn memory is NOT LangGraph's checkpointer - deliberately. A checkpointer
    restores the entire raw message history (every past tool call's full retrieved
    text) before every invoke(), which grows unbounded across a session. Instead, only
    the last HISTORY_TURNS turns' plain Q+A text is fed back (see _format_history()),
    same approach the old pipeline's intent_parser.py already used for its classifier.
    The system prompt explicitly tells the agent to judge from that text whether it
    already has enough to answer, or needs to search again - it is not forced into
    either path by a hard-coded rule."""
    history_text = _format_history(_history.get(thread_id, []))
    user_content = (
        f"Recent conversation:\n{history_text}\n\n"
        f"Current question: {query}"
    )

    registry: dict[str, dict] = {}
    tools = make_tools(registry)
    agent = create_agent(model=_get_model(), tools=tools, system_prompt=SYSTEM_PROMPT)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config={"recursion_limit": 8},
    )
    answer_text = result["messages"][-1].content
    citations = build_citations(registry)
    _record_history(thread_id, query, answer_text)

    # KNOWN LIMITATION: this is a rough heuristic, not a reliable refusal signal.
    # list_episodes() doesn't register anything (it's not chunk-based), so a valid
    # catalogue-only answer (or a topic-less recommendation's clarifying question)
    # will show refused=True here even though nothing was actually refused. Left as
    # informational only for this experimental testing pass - do not treat it as
    # trustworthy the way the fixed pipeline's `refused` field is.
    return {
        "query": query,
        "answer": answer_text,
        "citations": citations,
        "refused": len(citations) == 0,
        "tool_calls_this_turn": len(registry),
    }


def main():
    print("Tracer ReAct agent (experimental) - type 'quit' to exit.\n")
    thread_id = str(uuid.uuid4())

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit"):
            break

        result = run_agent(query, thread_id)
        print(f"\nTracer: {result['answer']}\n")
        if result["citations"]:
            print("Citations:")
            for c in result["citations"]:
                print(f"  - Episode {c['episode_number']}: {c['episode_title']} "
                      f"({generate.format_timestamp(c['start'])}-{generate.format_timestamp(c['end'])})")
        else:
            print("(no citations this turn)")
        print()


if __name__ == "__main__":
    main()
