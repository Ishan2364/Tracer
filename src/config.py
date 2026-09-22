"""Shared, named constants for the retrieval + generation pipeline (Phase 2+3)."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# LangSmith tracing - opt-in via .env (LANGSMITH_TRACING=true, LANGSMITH_API_KEY=...).
# Only set a default project name; never force tracing on for someone without a key.
os.environ.setdefault("LANGSMITH_PROJECT", "tracer")

# Collection / index
COLLECTION_NAME = "podcast_chunks"
INDEX_DIR = "index/chroma_db"
CHUNKS_DIR = "transcripts/chunks"
MANIFEST_PATH = "transcripts/episode_manifest.json"

# Embedding — query-time and index-build-time must use the exact same model,
# otherwise queries and chunks live in different vector spaces.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL_FALLBACK = "all-MiniLM-L6-v2"

# Retrieval
N_RESULTS = 5

# Best-match cosine similarity below this -> refuse rather than guess.
# This used to be a single hardcoded constant (0.56), hand-measured against our
# specific 4-episode catalogue's embedding distribution. That number doesn't
# transfer to a different set of episodes - a different corpus's vocabulary
# overlap shifts where "off-topic" actually plateaus in this embedding space.
#
# Instead, build_index.py now measures it automatically at index-build time:
# it embeds a fixed set of canary queries that are almost certainly NOT covered
# by any physics podcast (cooking, sports, geography, ...), checks their best
# similarity against whatever episodes actually got indexed, and writes
# ceiling + margin to CALIBRATION_PATH. That measured value is loaded below;
# DEFAULT_SIMILARITY_THRESHOLD is only a fallback for an index that predates
# calibration (no calibration.json yet) - run build_index.py to replace it.
DEFAULT_SIMILARITY_THRESHOLD = 0.56
CALIBRATION_PATH = "index/calibration.json"


def _load_similarity_threshold(default: float) -> float:
    path = Path(CALIBRATION_PATH)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return default


SIMILARITY_THRESHOLD = _load_similarity_threshold(DEFAULT_SIMILARITY_THRESHOLD)

# Generation
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "openai/gpt-oss-120b")
REFUSAL_MESSAGE = "The supplied episodes don't appear to cover this."
# Hard cap on total excerpt text placed in one prompt. Groq's on-demand tier rejects
# outright (not rate-limits) any single request over its per-model TPM cap - measured
# ~8000 tokens for openai/gpt-oss-120b - and a rebalanced broad query or an expanded
# follow-up can otherwise exceed that easily. ~4 chars/token, with headroom for the
# system prompt and completion.
MAX_EXCERPT_CHARS = 9000

# Intent parsing (Phase 4) — a small, cheap model, distinct from GENERATION_MODEL.
INTENT_MODEL = os.environ.get("INTENT_MODEL", "openai/gpt-oss-20b")

# Retrieval routing (Phase 4)
NAMED_COMPARISON_MAX = 12      # named_comparison with this many refs or fewer: one retrieve_scoped() call per
                                # episode. Was 6, with a "batch pairs into one Chroma call" scheme for 7-12 refs
                                # tried on top - dropped after a real test showed it silently lost episodes:
                                # a pair query pulls one shared top-N pool across both episodes, and if one
                                # scores more similar than the other, the weaker one can get zero chunks with
                                # no guaranteed floor. Chroma has no "top-N per episode within one call" option
                                # to fix that properly, so one call per episode - which does guarantee every
                                # named episode is represented - is simply extended to 12 instead. Chroma also
                                # runs as a local embedded client today, not a network service, so "fewer calls"
                                # has no real round-trip cost to justify the tradeoff anyway.
BROAD_N_RESULTS = 25           # unscoped n_results for broad/general/recommendation queries
PER_EPISODE_CAP = 3            # after a broad query, keep at most this many chunks per episode
FOLLOWUP_CONTEXT_WINDOW = 1    # chunks fetched on each side of a chunk during follow-up expansion
CONVERSATION_HISTORY_TURNS = 2 # turns of history shown to the intent parser
