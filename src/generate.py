"""Phase 3: prompt construction from retrieved chunks + the Groq generation call."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from groq import Groq
from langsmith import traceable

import config

SYSTEM_PROMPT = (
    "You are Tracer, a physics study companion. You answer only using the excerpts provided "
    "below, which are transcribed from a small podcast catalogue. Do not add outside "
    "knowledge, even if you know more about the topic.\n\n"
    "Rules:\n"
    "- Every factual claim you make must carry a citation in the form (Episode N, mm:ss-mm:ss), "
    "taken directly from the excerpt it is drawn from.\n"
    "- If the excerpts don't actually answer the question, say so explicitly rather than "
    "stretching them to fit.\n"
    "- Keep the answer grounded and concise.\n\n"
    "Tone: talk like a warm, encouraging teacher who's genuinely excited about this material - "
    "plain language, not a formal report. That warmth never loosens the rules above: it makes "
    "the explanation more human, it never lets you invent a fact, skip a citation, or paper "
    "over something the excerpts don't actually cover."
)

MULTI_EPISODE_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nThe excerpts below are grouped by episode. Address each episode that is named or "
    "relevant separately and explicitly - if a particular episode's excerpts don't actually "
    "cover the question, say so for that episode specifically rather than skipping it "
    "silently. After covering each episode individually, you may add a short synthesis "
    "comparing them - but do not blend episodes into one undifferentiated answer."
)

RECOMMENDATION_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nThe learner is asking WHICH episode(s) to listen to for a topic - they want a "
    "recommendation, not a summary of the topic itself. Explicitly recommend one specific "
    "episode (by number and title) as the best fit, and justify the choice using what's "
    "actually in the excerpts. If more than one episode genuinely covers the topic, name a "
    "primary recommendation first and mention the others as secondary options - don't just "
    "describe every episode's content evenly as if this were an open-ended question."
)

FOLLOWUP_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nThis is a follow-up to your previous answer. The excerpts below include a bit more "
    "surrounding context than the original ones did. Follow the learner's CURRENT request "
    "exactly, using only what it actually asks for - never assume it means 'explain slower "
    "with more detail'. If they ask you to go slower, break it into smaller steps, or walk "
    "through an example, do that. If they ask for something shorter, more concise, or a "
    "quick summary, make this answer SHORTER than your previous one, not longer. If they "
    "ask about one specific part, focus only on that part instead of re-explaining "
    "everything. Read the current request's own wording as the instruction - don't apply a "
    "default framing regardless of what was actually asked."
)


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, dict]:
    path = Path(config.MANIFEST_PATH)
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    lookup = {}
    for entry in manifest:
        lookup[entry["episode_id"]] = {
            "episode_number": entry["episode_number"],
            "title": derive_title(entry["source_file"]),
        }
    return lookup


def derive_title(source_file: str) -> str:
    stem = Path(source_file).stem
    match = re.match(r"^.*?\d+\s*[-:]\s*(.+)$", stem)
    return match.group(1).strip() if match else stem


def format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


# Matches "(Episode N, mm:ss-mm:ss)"-shaped citations the system prompt instructs the
# model to write - tolerant of the various dash-like characters actually observed in
# real outputs (plain hyphen, en dash, em dash, and the narrow non-breaking hyphen the
# model has been seen to use), and of a comma or colon after the episode number.
_CITATION_RE = re.compile(
    r"Episode\s+(\d+)[,:]?\s*(\d{1,3}):(\d{2})\s*[-‐‑‒–—]\s*(\d{1,3}):(\d{2})"
)

CITATION_MATCH_TOLERANCE_SECONDS = 3.0


def _parse_cited_ranges(answer_text: str) -> list[tuple[int, int, int]]:
    """Returns (episode_number, start_seconds, end_seconds) for every citation-shaped
    substring actually found in the model's answer text."""
    ranges = []
    for m in _CITATION_RE.finditer(answer_text):
        episode_number = int(m.group(1))
        start_seconds = int(m.group(2)) * 60 + int(m.group(3))
        end_seconds = int(m.group(4)) * 60 + int(m.group(5))
        ranges.append((episode_number, start_seconds, end_seconds))
    return ranges


def filter_citations_to_used(answer_text: str, chunks: list[dict]) -> list[dict]:
    """Narrows `chunks` (everything actually sent to the model - or, for the agent,
    everything any tool call returned this turn) down to only the ones the model's own
    answer text actually cites inline. Never trusts a new claim from the model - it only
    decides which subset of already-verified real chunk metadata to display, by matching
    the timestamps the model wrote against each chunk's real [start, end] range (with a
    small tolerance, since a citation naming a narrower sub-range within a chunk's true
    boundaries is a legitimate match, not a mismatch - see eval/cases.md's documented
    citation-precision quirk).

    Safety fallback: if zero citations can be parsed from the text at all (e.g. the
    model didn't follow the format), returns `chunks` unfiltered rather than an empty
    list - this can only ever narrow the result when there's real textual evidence to
    narrow it by, never produce a worse outcome than showing everything sent."""
    cited_ranges = _parse_cited_ranges(answer_text)
    if not cited_ranges:
        return chunks

    manifest = load_manifest()
    used = []
    for c in chunks:
        info = manifest.get(c["episode_id"], {})
        episode_number = info.get("episode_number", c.get("episode_number"))
        for cited_episode, cited_start, cited_end in cited_ranges:
            if cited_episode != episode_number:
                continue
            if c["start"] - CITATION_MATCH_TOLERANCE_SECONDS <= cited_start <= c["end"] + CITATION_MATCH_TOLERANCE_SECONDS:
                used.append(c)
                break
    return used


def format_chunk(chunk: dict) -> str:
    manifest = load_manifest()
    info = manifest.get(chunk["episode_id"], {"episode_number": "?", "title": chunk["episode_id"]})
    header = (
        f"[Episode {info['episode_number']}: {info['title']}, "
        f"{format_timestamp(chunk['start'])}-{format_timestamp(chunk['end'])}]"
    )
    return f"{header}\n{chunk['text']}"


def _fit_chunks_to_budget(chunks: list[dict], max_chars: int = config.MAX_EXCERPT_CHARS) -> list[dict]:
    """Keep chunks, in the given priority order, until the cumulative excerpt text
    would exceed max_chars - always keeps at least one chunk even if it alone is over."""
    kept = []
    total = 0
    for c in chunks:
        length = len(c["text"]) + 80  # + header overhead estimate
        if kept and total + length > max_chars:
            break
        kept.append(c)
        total += length
    return kept


def _unknown_episodes_note(unknown_episodes: list[int] | None) -> str:
    if not unknown_episodes:
        return ""
    refs = ", ".join(str(n) for n in unknown_episodes)
    return (
        f"\n\n(Note: the user also referenced episode(s) {refs}, which don't exist in "
        "this catalogue - mention this plainly rather than ignoring it.)"
    )


def build_messages(
    query: str, chunks: list[dict], unknown_episodes: list[int] | None = None
) -> tuple[list[dict], list[dict]]:
    """Returns (messages, chunks_sent) - chunks_sent is the post-budget-trim list that
    actually made it into the prompt, so callers can build citations from exactly what
    the model saw, not from the pre-trim set that may include chunks it never received."""
    chunks = _fit_chunks_to_budget(chunks)
    excerpts = "\n\n".join(format_chunk(c) for c in chunks)
    note = _unknown_episodes_note(unknown_episodes)
    user_prompt = f"Excerpts:\n\n{excerpts}{note}\n\nQuestion: {query}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, chunks


def build_messages_recommendation(
    query: str, chunks: list[dict], unknown_episodes: list[int] | None = None
) -> tuple[list[dict], list[dict]]:
    chunks = _fit_chunks_to_budget(chunks)
    excerpts = "\n\n".join(format_chunk(c) for c in chunks)
    note = _unknown_episodes_note(unknown_episodes)
    user_prompt = f"Excerpts:\n\n{excerpts}{note}\n\nQuestion: {query}"
    messages = [
        {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, chunks


def _group_chunks_by_episode(chunks: list[dict]) -> list[tuple[str, list[dict]]]:
    manifest = load_manifest()
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for c in chunks:
        eid = c["episode_id"]
        if eid not in groups:
            groups[eid] = []
            order.append(eid)
        groups[eid].append(c)
    order.sort(key=lambda eid: manifest.get(eid, {}).get("episode_number", 0))
    return [(eid, groups[eid]) for eid in order]


def format_excerpts_multi_episode(chunks: list[dict]) -> str:
    manifest = load_manifest()
    sections = []
    for eid, group in _group_chunks_by_episode(chunks):
        info = manifest.get(eid, {"episode_number": "?", "title": eid})
        lines = [f"=== Episode {info['episode_number']}: {info['title']} ==="]
        for c in sorted(group, key=lambda c: c["start"]):
            lines.append(f"[{format_timestamp(c['start'])}-{format_timestamp(c['end'])}]\n{c['text']}")
        sections.append("\n\n".join(lines))
    return "\n\n".join(sections)


def _fit_multi_episode_to_budget(chunks: list[dict], max_chars: int = config.MAX_EXCERPT_CHARS) -> list[dict]:
    """Same idea as _fit_chunks_to_budget, but allocates the budget per episode group
    first, so one episode's chunks can't crowd out another's in a multi-episode prompt."""
    groups = _group_chunks_by_episode(chunks)
    if not groups:
        return chunks
    per_group_budget = max_chars // len(groups)
    result = []
    for _, group in groups:
        result.extend(_fit_chunks_to_budget(group, max_chars=per_group_budget))
    return result


def build_messages_multi_episode(
    query: str, chunks: list[dict], unknown_episodes: list[int] | None = None
) -> tuple[list[dict], list[dict]]:
    chunks = _fit_multi_episode_to_budget(chunks)
    excerpts = format_excerpts_multi_episode(chunks)
    note = _unknown_episodes_note(unknown_episodes)
    user_prompt = f"Excerpts (grouped by episode):\n\n{excerpts}{note}\n\nQuestion: {query}"
    messages = [
        {"role": "system", "content": MULTI_EPISODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, chunks


def build_messages_followup(
    query: str, chunks: list[dict], previous_answer: str | None = None
) -> tuple[list[dict], list[dict]]:
    chunks = _fit_chunks_to_budget(chunks)
    excerpts = "\n\n".join(format_chunk(c) for c in chunks)
    prior = f"Your previous answer was:\n{previous_answer}\n\n" if previous_answer else ""
    user_prompt = f"{prior}Excerpts (expanded with surrounding context):\n\n{excerpts}\n\nFollow-up: {query}"
    messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, chunks


CHITCHAT_SYSTEM_PROMPT = """You are Tracer, a warm, encouraging physics study companion for a
small podcast catalogue. This message has been classified as social/meta, not a real content
question - a greeting, thanks, or a question about what you can help with. Respond briefly and
warmly, like a friendly teacher, in your own voice - no citations needed for this.

You are NOT permitted to answer any question about physics, the podcast episodes, or their
content using your own knowledge, even if it seems simple or is phrased casually. That
classification can be wrong - if this message actually contains or implies a real content
question (even a casual one, e.g. "can you explain black holes real quick"), recognize that
yourself and do not answer it - defer to a search instead.

Output STRICT JSON, nothing else:
{"reply": "...", "needs_retrieval": true|false}
- needs_retrieval: false - this really is social/meta. "reply" is your friendly response.
- needs_retrieval: true - this actually needs the episodes searched. "reply" should be a short,
  natural holding line (e.g. "Good question - let me check what the episodes say about that.");
  the real, grounded answer will be generated separately from a search.

Output ONLY the JSON object."""


def build_messages_chitchat(query: str, history: list[dict]) -> list[dict]:
    if history:
        lines = []
        for turn in history:
            lines.append(f"Q: {turn['query']}")
            lines.append(f"A: {turn['answer'][:200]}")
        history_text = "\n".join(lines)
    else:
        history_text = "(none - this is the first message)"
    user_prompt = f"Recent conversation:\n{history_text}\n\nMessage: {query}"
    return [
        {"role": "system", "content": CHITCHAT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


@traceable(run_type="llm", name="groq_generate")
def call_llm(messages: list[dict], model: str | None = None, response_format: dict | None = None) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to .env or export it in your environment.")

    client = Groq(api_key=api_key)
    kwargs = {"model": model or config.GENERATION_MODEL, "messages": messages, "temperature": 0.2}
    if response_format:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
