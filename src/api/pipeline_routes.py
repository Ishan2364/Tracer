"""Endpoints for building the index (Phases 1-2) from whatever audio is in /data.

Runs the existing, already-tested CLI scripts as subprocesses in a background
thread - transcribe.py -> chunk_transcripts.py -> build_index.py - rather than
re-implementing their logic, so this endpoint can never drift from what the
one-command CLI pipeline actually does. All three are idempotent, so calling
this endpoint again only processes newly-added audio files.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/api/ -> src/ -> project root
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_lock = threading.Lock()
_status: dict = {"state": "idle", "steps": [], "error": None}

STEPS = [
    ("transcribe", "src/transcribe.py"),
    ("chunk", "src/chunk_transcripts.py"),
    ("index", "src/build_index.py"),
]


def _run_pipeline() -> None:
    global _status
    with _lock:
        _status = {"state": "running", "steps": [], "error": None}

    for name, script in STEPS:
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        with _lock:
            _status["steps"].append({
                "step": name,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            })
        if result.returncode != 0:
            with _lock:
                _status["state"] = "failed"
                _status["error"] = f"'{name}' exited with code {result.returncode} - see steps[].stderr_tail"
            return

    with _lock:
        _status["state"] = "completed"
        _status["error"] = None


@router.post("/build")
def build_pipeline():
    """Kick off transcribe -> chunk -> embed in the background. Returns immediately -
    poll GET /pipeline/status for progress. Safe to call repeatedly: each underlying
    script skips audio/episodes it has already processed."""
    with _lock:
        if _status["state"] == "running":
            return {"status": "already_running"}
    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()
    return {"status": "started"}


@router.get("/status")
def pipeline_status():
    with _lock:
        return dict(_status)


@router.get("/health")
def pipeline_health():
    deepgram_key_set = bool(os.environ.get("DEEPGRAM_API_KEY"))
    data_dir_exists = DATA_DIR.is_dir()
    audio_file_count = 0
    if data_dir_exists:
        audio_file_count = sum(
            1 for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )

    with _lock:
        pipeline_state = _status["state"]

    ok = deepgram_key_set and data_dir_exists and audio_file_count > 0
    return {
        "status": "ok" if ok else "degraded",
        "deepgram_api_key_set": deepgram_key_set,
        "data_dir_exists": data_dir_exists,
        "audio_file_count": audio_file_count,
        "pipeline_state": pipeline_state,
    }
