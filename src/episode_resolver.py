"""Phase 4: map episode_refs (integers) to episode_id via the Phase 2 manifest."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import config


@lru_cache(maxsize=1)
def _load_number_to_id() -> dict[int, str]:
    with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {e["episode_number"]: e["episode_id"] for e in manifest}


def resolve_episodes(episode_refs: list[int]) -> tuple[list[dict], list[int]]:
    """Returns (resolved, unknown). `resolved` is a list of {episode_number, episode_id}
    in the order given; `unknown` is any referenced number not present in the manifest."""
    number_to_id = _load_number_to_id()
    resolved = []
    unknown = []
    for num in episode_refs:
        episode_id = number_to_id.get(num)
        if episode_id is None:
            unknown.append(num)
        else:
            resolved.append({"episode_number": num, "episode_id": episode_id})
    return resolved, unknown
