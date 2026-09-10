"""Phase 1: Audio -> Structured Transcript, via Deepgram Nova-3 (pre-recorded, diarized).

Usage:
    python src/transcribe.py [--input-dir /data] [--output-dir /transcripts] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_PARAMS = {
    "model": "nova-3",
    "diarize": "true",
    "punctuate": "true",
    "smart_format": "true",
    "utterances": "true",
}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}
CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}

MAX_RETRIES = 3
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def derive_episode_id(filename: str) -> str:
    stem = Path(filename).stem
    slug = stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def list_audio_files(input_dir: Path) -> list[Path]:
    files = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return files


def call_deepgram(audio_bytes: bytes, content_type: str, api_key: str) -> dict:
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": content_type,
    }

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                DEEPGRAM_URL,
                params=DEEPGRAM_PARAMS,
                headers=headers,
                data=audio_bytes,
                timeout=600,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Network error after {MAX_RETRIES} attempts: {exc}") from exc
            backoff = 2 ** attempt
            print(f"    network error ({exc}); retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(backoff)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in RETRY_STATUS_CODES:
            last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Deepgram API failed after {MAX_RETRIES} attempts: "
                    f"HTTP {resp.status_code}: {resp.text}"
                )
            backoff = 2 ** attempt
            print(f"    HTTP {resp.status_code}; retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(backoff)
            continue

        # Non-retryable error (e.g. 4xx auth/validation failure) - stop immediately.
        raise RuntimeError(
            f"Deepgram API returned non-retryable error HTTP {resp.status_code}:\n{resp.text}"
        )

    raise RuntimeError(f"Deepgram API failed: {last_exc}")


def parse_structured(raw: dict, episode_id: str, source_file: str) -> dict:
    results = raw.get("results", {})
    utterances_raw = results.get("utterances")

    if not utterances_raw:
        raise RuntimeError(
            f"'results.utterances' missing or empty for {source_file} — "
            "refusing to write an empty structured file."
        )

    duration = raw.get("metadata", {}).get("duration")

    utterances = []
    for i, u in enumerate(utterances_raw, start=1):
        speaker_idx = u.get("speaker", 0)
        utterances.append({
            "utterance_id": f"{episode_id}_u{i:04d}",
            "speaker": f"SPEAKER_{speaker_idx}",
            "start": u["start"],
            "end": u["end"],
            "text": u["transcript"],
            "confidence": u.get("confidence"),
        })

    structured = {
        "episode_id": episode_id,
        "source_file": source_file,
        "model": "nova-3",
        "duration_seconds": duration,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "utterances": utterances,
    }
    return structured


def get_actual_duration_seconds(path: Path) -> float | None:
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(path)
        if audio is not None and audio.info is not None:
            return float(audio.info.length)
    except Exception as exc:
        print(f"    warning: could not read duration via mutagen ({exc})")
    return None


def validate_structured(structured: dict, source_path: Path) -> list[str]:
    """Run validation checks. Raises RuntimeError on hard failures, returns list of warnings."""
    warnings: list[str] = []
    utterances = structured["utterances"]

    if not utterances:
        raise RuntimeError("utterances list is empty")

    prev_start = None
    for u in utterances:
        if u["start"] >= u["end"]:
            raise RuntimeError(f"utterance {u['utterance_id']} has start >= end ({u['start']} >= {u['end']})")
        if prev_start is not None and u["start"] < prev_start:
            raise RuntimeError(
                f"timestamps not monotonically non-decreasing at {u['utterance_id']} "
                f"(start {u['start']} < previous start {prev_start})"
            )
        prev_start = u["start"]

    reported_duration = structured.get("duration_seconds")
    actual_duration = get_actual_duration_seconds(source_path)
    if reported_duration is not None and actual_duration is not None:
        diff = abs(reported_duration - actual_duration)
        if diff > 5:
            warnings.append(
                f"duration mismatch: Deepgram reported {reported_duration:.1f}s, "
                f"actual file duration is {actual_duration:.1f}s (diff {diff:.1f}s)"
            )

    speakers = {u["speaker"] for u in utterances}
    if len(speakers) < 2:
        warnings.append(
            f"only {len(speakers)} distinct speaker(s) detected — fine if this episode "
            "is single-narrator style, otherwise check diarization."
        )

    return warnings


def process_episode(
    audio_path: Path,
    output_dir: Path,
    api_key: str,
    force: bool,
) -> dict | None:
    episode_id = derive_episode_id(audio_path.name)
    raw_path = output_dir / "raw" / f"{episode_id}.json"
    structured_path = output_dir / "structured" / f"{episode_id}.json"

    if structured_path.exists() and not force:
        print(f"[{episode_id}] already transcribed, skipping (use --force to re-run)")
        with open(structured_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {
            "episode_id": episode_id,
            "duration": existing.get("duration_seconds"),
            "num_utterances": len(existing.get("utterances", [])),
            "num_speakers": len({u["speaker"] for u in existing.get("utterances", [])}),
            "skipped": True,
        }

    ext = audio_path.suffix.lower()
    content_type = CONTENT_TYPES[ext]

    print(f"[{episode_id}] transcribing {audio_path.name} ...")
    audio_bytes = audio_path.read_bytes()

    raw = call_deepgram(audio_bytes, content_type, api_key)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    structured = parse_structured(raw, episode_id, audio_path.name)

    warnings = validate_structured(structured, audio_path)
    for w in warnings:
        print(f"    warning: {w}")

    structured_path.parent.mkdir(parents=True, exist_ok=True)
    with open(structured_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2)

    num_speakers = len({u["speaker"] for u in structured["utterances"]})
    print(
        f"[{episode_id}] done — duration={structured['duration_seconds']}s, "
        f"utterances={len(structured['utterances'])}, speakers={num_speakers}"
    )

    return {
        "episode_id": episode_id,
        "duration": structured["duration_seconds"],
        "num_utterances": len(structured["utterances"]),
        "num_speakers": num_speakers,
        "skipped": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Transcribe podcast audio via Deepgram Nova-3.")
    parser.add_argument("--input-dir", default="data", help="Directory containing raw audio files")
    parser.add_argument("--output-dir", default="transcripts", help="Directory to write transcripts into")
    parser.add_argument("--force", action="store_true", help="Re-transcribe even if structured output exists")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("ERROR: DEEPGRAM_API_KEY not set. Add it to .env or export it in your environment.", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "structured").mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"ERROR: input dir {input_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    audio_files = list_audio_files(input_dir)
    if not audio_files:
        print(f"No audio files found in {input_dir} (looked for {sorted(AUDIO_EXTENSIONS)}).")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio file(s) in {input_dir}.\n")

    results = []
    failures = []
    for audio_path in audio_files:
        try:
            result = process_episode(audio_path, output_dir, api_key, args.force)
            if result is not None:
                results.append(result)
        except Exception as exc:
            print(f"[{derive_episode_id(audio_path.name)}] FAILED: {exc}", file=sys.stderr)
            failures.append((audio_path.name, str(exc)))

    print("\n=== Summary ===")
    header = f"{'episode_id':<35} {'duration(s)':>12} {'#utterances':>12} {'#speakers':>10} {'status':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "skipped" if r["skipped"] else "ok"
        duration = r["duration"] if r["duration"] is not None else "?"
        print(f"{r['episode_id']:<35} {duration!s:>12} {r['num_utterances']:>12} {r['num_speakers']:>10} {status:>10}")

    if failures:
        print("\n=== Failures ===")
        for name, err in failures:
            print(f"{name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
