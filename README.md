<div align="center">

<img src="assets/banner.svg" alt="Tracer" width="100%" />

<br/>

<img src="https://img.shields.io/badge/Python-3.10%2B-1e3a8a?style=flat-square" alt="Python 3.10+" />
<img src="https://img.shields.io/badge/Speech-Deepgram%20Nova--3-0e7490?style=flat-square" alt="Deepgram Nova-3" />
<img src="https://img.shields.io/badge/LLM-Groq-7c3aed?style=flat-square" alt="Groq" />
<img src="https://img.shields.io/badge/Vector%20DB-Chroma-b45309?style=flat-square" alt="Chroma" />
<img src="https://img.shields.io/badge/API-FastAPI-166534?style=flat-square" alt="FastAPI" />

</div>

<br/>

**Tracer is a conversational study companion for a small physics podcast catalogue.** A
learner asks questions in plain language; every answer traces back to an exact timestamp
in the actual audio, and the product says so plainly when something isn't covered
instead of guessing. See [`PRODUCT_NOTE.md`](PRODUCT_NOTE.md) for who it's for and why
it's scoped the way it is.

<br/>

## How it works

<img src="assets/pipeline.svg" alt="Audio to structured transcript pipeline" width="100%" />

**Offline, once per episode:** each audio file goes to Deepgram's **Nova-3** model in a
single call with diarization enabled (diagram above). The raw response is saved
verbatim, normalized into a per-utterance schema, and validated. Utterances are then
merged into ~30–90s conversational chunks and embedded locally (`bge-base-en-v1.5`)
into a persistent Chroma collection (cosine distance).

**Live, every turn:** a query is classified by a small model into one of several
intents (single-episode, multi-episode comparison, catalogue-wide, recommendation,
follow-up, chitchat, or a catalogue/inventory question), routed to the matching
retrieval strategy, and — for anything that's actually a content question — answered by
a larger model that must cite `(Episode N, mm:ss–mm:ss)` for every claim. A
[published architecture diagram](architecture_diagram_spec.md) walks through the full
decision tree (routing, the refusal gate, the two-gate chitchat safety net) in detail.

<br/>

## Features

**Grounded Q&A**
- Every claim cited as `(Episode N, mm:ss–mm:ss)`, built from retrieval metadata — never parsed out of free text, so a citation can't be hallucinated independently of what was actually retrieved
- Honest refusal, calibrated automatically per index build against known off-topic "canary" queries rather than a hand-picked number
- Single-episode, multi-episode comparison (including partial-coverage handling and disambiguating ambiguous title references like "the relativity episode"), catalogue-wide, and episode-recommendation queries — each routed differently

**Conversation & memory**
- Multi-turn follow-ups that build on the prior answer's exact grounding, adapting to what's actually asked (more detail vs. more concise) rather than assuming
- Episodes referenced by number or by title/topic
- Session-based memory — conversation state persists across separate calls via `--session-id` (CLI) or `session_id` (API), not just one process
- Time-range-scoped questions ("what's covered between minute 3 and 18"), enforced at retrieval, not just claimed in the answer
- Deterministic catalogue/inventory answers ("what episodes do you have") — read from the manifest directly, never sampled from retrieval, so the list is always complete and consistent

**Trust & safety**
- Nonexistent episode references are called out explicitly, never silently dropped
- Chitchat (greetings, thanks, "what can you do") is handled warmly without touching retrieval — but is structurally unable to answer real content questions from its own knowledge; falls back to a real grounded search if misclassified

**Interfaces**
- CLI (`src/chat.py`): interactive loop, single-shot `--query`, `--debug` routing visibility
- FastAPI wrapper (`src/api/`): endpoints for building the index and for chat, each with a real health check — see [Running the API](#running-the-api) below
- Full evaluation harness with hard (deterministic) and LLM-judge checks — see [`EVAL.md`](EVAL.md)

<br/>

## Dataset

8 episodes transcribed, chunked, and indexed:

| Episode | Duration | Chunks |
|---|---|---|
| 1 — Einstein's Special Relativity | 52m 14s | 36 |
| 2 — How Black Holes Radiate (Hawking, 1975) | 45m 16s | 31 |
| 3 — The Double Helix (Watson & Crick, 1953) | 43m 16s | 30 |
| 4 — Shannon and the Birth of Information (1948) | 39m 52s | 28 |
| 5 — Attention Is All You Need (2017) | 36m 24s | 25 |
| 6 — General Relativity (Einstein, 1915) | 33m 48s | 24 |
| 7 — On Computable Numbers (Turing, 1936) | 33m 2s | 23 |
| 8 — Natural Selection (Darwin & Wallace, 1858) | 33m 13s | 23 |
| **Total** | **~5h 17m** | **220** |

<br/>

## One-time setup

**Prerequisites:** Python 3.10+, a [Deepgram](https://deepgram.com) API key, a
[Groq](https://groq.com) API key. A [LangSmith](https://smith.langchain.com) key is
optional (tracing only).

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

`.env`:
```
DEEPGRAM_API_KEY=your_deepgram_api_key_here
GROQ_API_KEY=your_groq_api_key_here

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=            # optional - leave blank to disable tracing
LANGSMITH_PROJECT=tracer
```

Build the index from whatever audio is in `/data` — one command per stage, all
idempotent (safe to re-run; only new/changed episodes are processed, at zero extra
API cost):

```bash
python src/transcribe.py          # audio -> raw + structured transcripts (Deepgram)
python src/chunk_transcripts.py   # transcripts -> merged chunks + episode manifest
python src/build_index.py         # chunks -> embedded, into the Chroma index
```

That's the whole one-time setup. Re-run all three any time you add new audio files to
`/data` — each one skips work it's already done.

<br/>

## Running it

### Chat (CLI)

```bash
python src/chat.py                                  # interactive, multi-turn
python src/chat.py --query "..."                    # single-shot
python src/chat.py --session-id abc --query "..."   # persists to sessions/abc.json -
                                                      # a later call with the same id resumes it
python src/chat.py --debug                           # also print intent/routing info
```

### API (for a frontend)

Start the server from the project root (paths inside the app are relative to it, same
as every CLI script):

```bash
uvicorn src.api.app:app --reload --port 8000
```

Interactive docs (try every endpoint from the browser) at `http://localhost:8000/docs`.

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | basic liveness |
| GET | `/pipeline/health` | Deepgram key set, `/data` present, current pipeline state |
| POST | `/pipeline/build` | runs the 3-stage ingest pipeline in the background |
| GET | `/pipeline/status` | poll progress of the last/current build |
| GET | `/chat/health` | Groq key set, indexed chunk count, calibrated threshold |
| POST | `/chat` | `{"session_id": "...", "query": "..."}` → grounded answer + citations |

**Typical frontend flow:**
1. On load, call `GET /chat/health` — if `indexed_chunk_count` is 0 or the status isn't
   `"ok"`, prompt to run the pipeline first (or call `POST /pipeline/build` yourself and
   poll `GET /pipeline/status` until it says `"completed"`).
2. Generate a `session_id` per browser tab/user (any string — a UUID is fine) and reuse
   it on every `POST /chat` call from that session so follow-ups and memory work.
3. Render `answer` as the reply and `citations` as clickable source chips (each has
   `episode_number`, `episode_title`, `start`, `end` in seconds — enough to jump to that
   point if you have the audio).

**Example — request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "query": "What did Einstein realize about simultaneity?"}'
```

**Response:**
```json
{
  "query": "What did Einstein realize about simultaneity?",
  "answer": "Einstein realized that simultaneity is relative... (Episode 1, 46:28-47:55)",
  "citations": [
    {"chunk_id": "...", "episode_number": 1, "episode_title": "Einstein's Special Relativity", "start": 2875.25, "end": 2963.71}
  ],
  "query_type": "single_episode",
  "retrieval_mode": "scoped_loop",
  "refused": false,
  "unknown_episodes": []
}
```

CORS is wide open (`allow_origins=["*"]`) for local frontend development — this is a
single-user local tool per the project's scope (see `PRODUCT_NOTE.md`'s non-goals), so
tighten it if this is ever exposed beyond localhost.

### Evaluation

```bash
python src/eval/run_eval.py --run-id baseline              # Phase 5 harness, 12 cases
python src/eval/run_eval.py --run-id baseline --with-judge  # + LLM-judge scoring
python src/eval/compare_runs.py --baseline baseline --candidate v2
```

See [`EVAL.md`](EVAL.md) for success criteria, results, and the baseline→improvement
comparison; `eval/eval_v2_cases.md` for a 15-case manual-review round covering every
feature above.

<br/>

## Output

```
transcripts/
  raw/{episode_id}.json          verbatim Deepgram API response
  structured/{episode_id}.json   normalized transcript (per-utterance)
  chunks/{episode_id}.json       merged, embeddable chunks
  episode_manifest.json          episode_number ↔ episode_id ↔ title lookup
index/
  chroma_db/                     persistent Chroma collection (cosine, 768-dim)
  calibration.json               auto-calibrated refusal threshold
sessions/
  {session_id}.json              persisted conversation state per session
```

Structured transcript schema:

```json
{
  "episode_id": "great_papers_01_einstein_s_special_relativity",
  "source_file": "Great Papers 01 - Einstein's Special Relativity.mp3",
  "model": "nova-3",
  "duration_seconds": 3134.01,
  "generated_at": "2026-09-09T21:48:32Z",
  "utterances": [
    {
      "utterance_id": "great_papers_01_einstein_s_special_relativity_u0002",
      "speaker": "SPEAKER_0",
      "start": 2.0,
      "end": 7.12,
      "text": "A 26 year old is examining patent applications at a desk in Bern,",
      "confidence": 0.9817
    }
  ]
}
```

A chat response:

```json
{
  "query": "What did Einstein realize about simultaneity?",
  "answer": "...(Episode 1, 46:28-47:55)",
  "citations": [{"episode_number": 1, "episode_title": "Einstein's Special Relativity", "start": 2875.25, "end": 2963.71}],
  "query_type": "single_episode",
  "retrieval_mode": "scoped_loop",
  "refused": false,
  "unknown_episodes": []
}
```

<br/>

## Validation

Every structured transcript is checked automatically after transcription:

| Check | What it catches |
|---|---|
| Non-empty utterance list | An episode that produced no transcript at all |
| Monotonic timestamps | Out-of-order segments from a parsing bug |
| `start < end` on every utterance | Malformed time ranges |
| Duration cross-check | Reported duration vs. actual audio file length (via `mutagen`) |
| Speaker coverage | Warns if diarization returned only one speaker |

The refusal threshold is validated the same way, but at index-build time: `build_index.py`
measures how high a set of definitely-off-topic "canary" queries score against the
actual indexed episodes, and sets the threshold just above that measured ceiling —
see `index/calibration.json` after a build.

<br/>

## Project structure

```
.
├── assets/
├── data/
├── transcripts/
├── index/
├── sessions/
├── eval/
│   ├── cases.json
│   ├── eval_v2_cases.md
│   ├── EXPECTED_ANSWERS.md
│   ├── results/
│   └── results_v2/
├── src/
│   ├── transcribe.py
│   ├── chunk_transcripts.py
│   ├── build_index.py
│   ├── config.py
│   ├── intent_parser.py
│   ├── episode_resolver.py
│   ├── retrieval_router.py
│   ├── retrieve.py
│   ├── generate.py
│   ├── answer.py
│   ├── conversation_state.py
│   ├── session_store.py
│   ├── chat.py
│   ├── api/
│   │   ├── app.py
│   │   ├── pipeline_routes.py
│   │   └── chat_routes.py
│   └── eval/
│       ├── run_eval.py
│       ├── run_eval_v2.py
│       ├── scoring_hard.py
│       ├── scoring_judge.py
│       └── compare_runs.py
├── .env.example
├── requirements.txt
├── PRODUCT_NOTE.md
├── EVAL.md
└── README.md
```

**Generated at runtime, not hand-edited** (`data/` is the one exception — read-only input you supply):

| Path | Contents |
|---|---|
| `data/` | Raw audio input — read-only |
| `transcripts/` | Phase 1–2 output: raw/structured transcripts, chunks, episode manifest |
| `index/` | Chroma collection + the auto-calibrated refusal threshold |
| `sessions/` | Persisted conversation state, one file per `session_id` |
| `eval/results/`, `eval/results_v2/` | Raw + scored output from evaluation runs |

**`src/` — what each module owns:**

| File | Role |
|---|---|
| `transcribe.py` | Phase 1 — audio → structured transcript (Deepgram) |
| `chunk_transcripts.py` | Phase 2 — transcript → merged chunks + episode manifest |
| `build_index.py` | Phase 2 — chunks → Chroma index, plus refusal-threshold calibration |
| `config.py` | Shared constants: models, thresholds, paths |
| `intent_parser.py` | Classifies each query (single-episode, comparison, chitchat, catalogue, ...) |
| `episode_resolver.py` | Resolves `episode_refs` → `episode_id`, flags unknown episodes |
| `retrieval_router.py` | Routes a classified intent to the matching retrieval strategy |
| `retrieve.py` | The actual Chroma queries — scoped, broad, time-range-filtered |
| `generate.py` | Prompt construction (all variants) + the Groq generation call |
| `answer.py` | Orchestrator — `answer_conversational()` ties everything together |
| `conversation_state.py` | In-memory per-turn state (history, last grounding, last answer) |
| `session_store.py` | Persists/loads `ConversationState` by `session_id` |
| `chat.py` | CLI entrypoint |
| `api/` | FastAPI wrapper — `app.py` (server), `pipeline_routes.py`, `chat_routes.py` |
| `eval/` | Evaluation harness — runners (`run_eval.py`, `run_eval_v2.py`) and scoring |
