"""Shared, named constants for the retrieval + generation pipeline (Phase 2+3)."""

import os

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
# Phase 3's placeholder (0.35) never separated in-/out-of-scope queries with
# bge-base-en-v1.5 on this catalogue, so it was raised to 0.58 (off-topic queries
# scored up to 0.52, on-topic queries started at 0.67).
# Phase 5 eval (case_12) found 0.58 still too high: a vague-but-legitimately-broad
# query ("tell me something interesting") scored 0.5775 - just under the line -
# and was incorrectly refused, while the genuinely uncovered case in the same run
# scored 0.5491. 0.56 sits in the empirical gap between those two measured points.
SIMILARITY_THRESHOLD = 0.56

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
NAMED_COMPARISON_MAX = 6       # named_comparison with more refs than this is treated as broad
BROAD_N_RESULTS = 25           # unscoped n_results for broad/general/recommendation queries
PER_EPISODE_CAP = 3            # after a broad query, keep at most this many chunks per episode
FOLLOWUP_CONTEXT_WINDOW = 1    # chunks fetched on each side of a chunk during follow-up expansion
CONVERSATION_HISTORY_TURNS = 2 # turns of history shown to the intent parser
