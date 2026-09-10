"""Phase 5 - LLM-judge checks, only for what a hard check genuinely can't decide.

One combined judge call per case (not per-citation) to keep API cost down. The judge
is given only the specific retrieved excerpt(s) and the answer - never the full
transcript - per the spec's "too much context rates generously" warning.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from groq import APIError, Groq

import config

JUDGE_MODEL = config.GENERATION_MODEL  # claim-support judgment needs real reasoning quality
MAX_RETRIES = 2

JUDGE_SYSTEM_PROMPT = """You are a strict grading judge for a podcast Q&A system. You are given:
- the user's query
- one or more source excerpts that were actually retrieved and shown to the system
- the system's generated answer
- which episode number(s), if any, the answer should explicitly acknowledge as NOT covering the topic

Judge ONLY against the excerpts given - do not use outside knowledge of the subject.

Output STRICT JSON, nothing else, matching this schema:
{
  "claim_support": {"supported": true|false, "rationale": "one line"},
  "episode_acknowledgment": {"acknowledged": true|false|null, "rationale": "one line"},
  "helpfulness": {"rating": 1-5, "rationale": "one line"}
}

claim_support: does the excerpt text actually support the factual claims the answer makes
about that episode? false if the answer invents or stretches something not in the excerpt.

episode_acknowledgment: null if no episode needed acknowledgment for this case. Otherwise
true only if the answer EXPLICITLY states that the given episode(s) don't cover the topic
(or don't exist) - false if it silently omits them or gives a vague non-answer.

helpfulness: 1-5, would a physics-curious learner with no formal background find this
answer clear and genuinely useful (not just technically grounded)?"""


@lru_cache(maxsize=None)
def _load_episode_chunks(episode_id: str) -> tuple:
    path = Path(config.CHUNKS_DIR) / f"{episode_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return tuple(json.load(f))


def _chunk_text(chunk_id: str, episode_id: str) -> str | None:
    for c in _load_episode_chunks(episode_id):
        if c["chunk_id"] == chunk_id:
            return c["text"]
    return None


def _episode_id_from_chunk_id(chunk_id: str) -> str | None:
    # chunk_id is "{episode_id}_c####" - strip the trailing "_c####" suffix.
    return chunk_id.rsplit("_c", 1)[0] if "_c" in chunk_id else None


def _representative_excerpts(turns_raw: list[dict], max_excerpts: int = 4) -> str:
    """One excerpt per distinct episode cited in the final turn, capped for prompt size."""
    seen_episodes = set()
    blocks = []
    for c in turns_raw[-1]["citations"]:
        if c["episode_number"] in seen_episodes:
            continue
        episode_id = _episode_id_from_chunk_id(c["chunk_id"])
        text = _chunk_text(c["chunk_id"], episode_id) if episode_id else None
        if text is None:
            continue
        seen_episodes.add(c["episode_number"])
        blocks.append(f"[Episode {c['episode_number']}, {c['chunk_id']}]\n{text}")
        if len(blocks) >= max_excerpts:
            break
    return "\n\n".join(blocks)


def _episodes_to_acknowledge(case: dict) -> list[int]:
    uncovered = [int(k) for k, v in case["coverage_expectation"].items() if v is False]
    unknown = case.get("expected_unknown_episodes", [])
    return sorted(set(uncovered) | set(unknown))


def score_case(case: dict, turns_raw: list[dict]) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY not set - judge scoring skipped"}

    excerpts = _representative_excerpts(turns_raw)
    to_ack = _episodes_to_acknowledge(case)
    final_answer = turns_raw[-1]["answer"]
    final_query = turns_raw[-1]["query"]

    ack_instruction = (
        f"Episode(s) that should be explicitly acknowledged as not covering the topic "
        f"(or not existing): {to_ack}" if to_ack else
        "No episode needs acknowledgment for this case (episode_acknowledgment should be null)."
    )

    user_prompt = (
        f"Query: {final_query}\n\n"
        f"Retrieved excerpts:\n\n{excerpts or '(none retrieved)'}\n\n"
        f"System's answer:\n{final_answer}\n\n"
        f"{ack_instruction}"
    )

    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for _ in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except (APIError, json.JSONDecodeError) as exc:
            last_error = exc
            messages.append({
                "role": "user",
                "content": f"That failed ({exc}). Re-output ONLY the JSON object matching the schema.",
            })

    return {"error": f"judge failed after retries: {last_error}"}
