"""Phase 4: classify each incoming query before retrieval runs.

Uses a small, cheap model (config.INTENT_MODEL) - distinct from the heavier
generation model - with strict JSON output, validated and retried on malformed output.
"""

from __future__ import annotations

import json
import os

from groq import APIError, Groq
from langsmith import traceable

import config
from generate import load_manifest

VALID_QUERY_TYPES = {
    "single_episode",
    "named_comparison",
    "broad_comparison",
    "recommendation",
    "follow_up",
    "general",
}

MAX_RETRIES = 2

SYSTEM_PROMPT = """You are an intent classifier for a physics podcast Q&A system.

Given the user's current query and the last few turns of conversation, output STRICT JSON
matching exactly this schema, nothing else:

{"query_type": "single_episode|named_comparison|broad_comparison|recommendation|follow_up|general", "topic": "short phrase", "episode_refs": [1, 2]}

query_type definitions:
- single_episode: unambiguously names exactly one episode.
- named_comparison: names a small explicit set of episodes (2-6). ALSO use this when a
  title/topic reference is ambiguous - i.e. it could plausibly match more than one
  episode in the Known episodes list (e.g. "the relativity episode" when the list has
  both "Special Relativity" and "General Relativity") - list ALL plausible matches in
  episode_refs rather than guessing just one. Only resolve to a single episode when the
  query's own wording clearly distinguishes it from the others (e.g. "the SPECIAL
  relativity episode" is unambiguous even with two relativity episodes present).
- broad_comparison: asks to compare/cover "all episodes", or names no specific episodes
  but implies more than a handful.
- recommendation: asks which episode to listen to for a topic.
- follow_up: introduces no new topic and refers back to what was just discussed
  (e.g. "walk me through it", "I didn't get that example", "what do you mean",
  "go on", "can you explain that differently").
- general: a topic question with no episode scoping implied either way.

episode_refs: a flat list of integers, using the episode numbers from the "Known
episodes" list provided below the query. Resolve BOTH explicit numeric references
("episode 1", "episodes 1, 2, 3", "episodes 1-3" -> [1, 2, 3]) AND references made by
title or topic instead of number (e.g. "the black holes episode", "the one about DNA",
"the Shannon episode") - match those against the titles in the Known episodes list and
resolve to the correct number(s). When a title/topic reference matches more than one
episode ambiguously, include all of them (see named_comparison above) rather than
arbitrarily picking one. Empty list if no episode is named or clearly implied by
title/topic.

topic: a short phrase suitable for embedding-based search - strip out episode
references and conversational framing. For a follow_up, topic may be empty.

Output ONLY the JSON object."""


def _validate(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    if data.get("query_type") not in VALID_QUERY_TYPES:
        raise ValueError(f"query_type must be one of {sorted(VALID_QUERY_TYPES)}, got {data.get('query_type')!r}")
    if "topic" not in data or not isinstance(data["topic"], str):
        raise ValueError("topic must be a string")
    refs = data.get("episode_refs")
    if not isinstance(refs, list) or not all(isinstance(r, int) for r in refs):
        raise ValueError("episode_refs must be a list of integers")
    return {
        "query_type": data["query_type"],
        "topic": data["topic"],
        "episode_refs": refs,
    }


def _episode_list_text() -> str:
    manifest = load_manifest()
    entries = sorted(manifest.values(), key=lambda e: e["episode_number"])
    return "\n".join(f"{e['episode_number']}: {e['title']}" for e in entries)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none - this is the first turn)"
    lines = []
    for turn in history:
        lines.append(f"Q: {turn['query']}")
        lines.append(f"A: {turn['answer'][:300]}")
    return "\n".join(lines)


@traceable(run_type="llm", name="parse_intent")
def parse_intent(query: str, history: list[dict] | None = None) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to .env or export it in your environment.")

    client = Groq(api_key=api_key)
    history = history or []

    user_content = (
        f"Known episodes:\n{_episode_list_text()}\n\n"
        f"Conversation history:\n{_format_history(history)}\n\n"
        f"Current query: {query}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=config.INTENT_MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except APIError as exc:
            # Groq's JSON mode occasionally rejects its own generation outright
            # (empty failed_generation, json_validate_failed) - retry with a nudge
            # rather than letting it crash the whole turn.
            last_error = exc
            messages.append({
                "role": "user",
                "content": "That failed to generate valid JSON. Re-output ONLY the JSON object matching the schema, with no extra text.",
            })
            continue

        raw = response.choices[0].message.content
        try:
            return _validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That was invalid: {exc}. Re-output ONLY the JSON object matching the schema.",
            })

    raise RuntimeError(f"Intent parser failed to produce valid JSON after {MAX_RETRIES + 1} attempts: {last_error}")
