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


def build_messages(query: str, chunks: list[dict], unknown_episodes: list[int] | None = None) -> list[dict]:
    chunks = _fit_chunks_to_budget(chunks)
    excerpts = "\n\n".join(format_chunk(c) for c in chunks)
    note = _unknown_episodes_note(unknown_episodes)
    user_prompt = f"Excerpts:\n\n{excerpts}{note}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


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
) -> list[dict]:
    chunks = _fit_multi_episode_to_budget(chunks)
    excerpts = format_excerpts_multi_episode(chunks)
    note = _unknown_episodes_note(unknown_episodes)
    user_prompt = f"Excerpts (grouped by episode):\n\n{excerpts}{note}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": MULTI_EPISODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_messages_followup(query: str, chunks: list[dict], previous_answer: str | None = None) -> list[dict]:
    chunks = _fit_chunks_to_budget(chunks)
    excerpts = "\n\n".join(format_chunk(c) for c in chunks)
    prior = f"Your previous answer was:\n{previous_answer}\n\n" if previous_answer else ""
    user_prompt = f"{prior}Excerpts (expanded with surrounding context):\n\n{excerpts}\n\nFollow-up: {query}"
    return [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


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
