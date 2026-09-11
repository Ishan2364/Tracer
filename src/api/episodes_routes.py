"""Endpoints for the episode catalogue - list episodes and stream their raw audio,
so a frontend can display and play them without touching the filesystem directly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parent.parent  # src/api/ -> src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from generate import derive_title  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}

router = APIRouter(prefix="/episodes", tags=["episodes"])


class Episode(BaseModel):
    episode_number: int
    episode_id: str
    title: str
    source_file: str
    duration_seconds: float | None = None


def _load_manifest() -> list[dict]:
    manifest_path = Path(config.MANIFEST_PATH)
    if not manifest_path.exists():
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("", response_model=list[Episode])
def list_episodes():
    manifest = sorted(_load_manifest(), key=lambda e: e["episode_number"])
    return [
        Episode(
            episode_number=e["episode_number"],
            episode_id=e["episode_id"],
            title=derive_title(e["source_file"]),
            source_file=e["source_file"],
            duration_seconds=e.get("duration_seconds"),
        )
        for e in manifest
    ]


@router.get("/{episode_number}/audio")
def get_episode_audio(episode_number: int):
    manifest = _load_manifest()
    entry = next((e for e in manifest if e["episode_number"] == episode_number), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_number} not found")

    audio_path = DATA_DIR / entry["source_file"]
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file for episode {episode_number} not found in /data")

    content_type = CONTENT_TYPES.get(audio_path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=str(audio_path),
        media_type=content_type,
        filename=audio_path.name,
        content_disposition_type="inline",  # play in <audio>, don't trigger a download
    )
