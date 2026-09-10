"""Phase 2 (step 1): merge Phase 1 utterances into coherent, embeddable chunks.

Usage:
    python src/chunk_transcripts.py [--input-dir transcripts/structured] [--output-dir transcripts] [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MAX_CHUNK_SECONDS = 90
MIN_CHUNK_SECONDS = 15
SILENCE_GAP_SECONDS = 4

VALID_DURATION_MIN = 10
VALID_DURATION_MAX = 95


def group_utterances(utterances: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []

    for u in utterances:
        if not current:
            current = [u]
            continue

        prev = current[-1]
        gap = u["start"] - prev["end"]
        duration_if_added = u["end"] - current[0]["start"]
        close_trigger = gap > SILENCE_GAP_SECONDS or duration_if_added > MAX_CHUNK_SECONDS

        if close_trigger:
            current_duration = current[-1]["end"] - current[0]["start"]
            if current_duration < MIN_CHUNK_SECONDS:
                # Too short to stand alone - keep absorbing utterances instead of
                # emitting a low-context fragment.
                current.append(u)
            else:
                groups.append(current)
                current = [u]
        else:
            current.append(u)

    if current:
        groups.append(current)

    return groups


def build_chunk(episode_id: str, group: list[dict], index: int) -> dict:
    speakers_seen: list[str] = []
    for u in group:
        if u["speaker"] not in speakers_seen:
            speakers_seen.append(u["speaker"])

    if len(speakers_seen) > 1:
        text = " ".join(f"{u['speaker']}: {u['text']}" for u in group)
    else:
        text = " ".join(u["text"] for u in group)

    return {
        "chunk_id": f"{episode_id}_c{index:04d}",
        "episode_id": episode_id,
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "speakers": ",".join(speakers_seen),
        "utterance_ids": ",".join(u["utterance_id"] for u in group),
        "text": text,
    }


def validate_chunks(episode_id: str, chunks: list[dict], duration_seconds: float | None) -> list[str]:
    warnings: list[str] = []

    if not chunks:
        raise RuntimeError(f"{episode_id}: chunking produced zero chunks")

    prev_end = None
    for c in chunks:
        if not c["text"].strip():
            raise RuntimeError(f"{episode_id}: chunk {c['chunk_id']} has empty text")
        if c["start"] >= c["end"]:
            raise RuntimeError(f"{episode_id}: chunk {c['chunk_id']} has start >= end")
        if prev_end is not None and c["start"] < prev_end:
            raise RuntimeError(
                f"{episode_id}: chunk {c['chunk_id']} overlaps previous chunk "
                f"(start {c['start']} < previous end {prev_end})"
            )
        prev_end = c["end"]

        dur = c["end"] - c["start"]
        if not (VALID_DURATION_MIN <= dur <= VALID_DURATION_MAX):
            warnings.append(
                f"{episode_id}: chunk {c['chunk_id']} duration {dur:.1f}s outside "
                f"[{VALID_DURATION_MIN}, {VALID_DURATION_MAX}]s"
            )

    if duration_seconds:
        hours = duration_seconds / 3600
        expected_min = 40 * hours
        expected_max = 120 * hours
        if not (expected_min <= len(chunks) <= expected_max):
            warnings.append(
                f"{episode_id}: {len(chunks)} chunks for {hours:.2f}h of audio "
                f"(expected roughly {expected_min:.0f}-{expected_max:.0f})"
            )

    return warnings


def derive_episode_number(source_files: list[str]) -> tuple[dict[str, int], str]:
    """Returns {source_file: episode_number} and a note on the method used."""
    explicit: dict[str, int] = {}
    for name in source_files:
        stem = Path(name).stem
        tokens = re.split(r"[^0-9A-Za-z]+", stem)
        for tok in tokens[:4]:
            if tok.isdigit() and len(tok) <= 3:
                explicit[name] = int(tok)
                break

    if len(explicit) == len(source_files):
        return explicit, "explicit episode numbers parsed from filenames"

    positional = {name: i + 1 for i, name in enumerate(sorted(source_files))}
    return positional, "positional order (sorted filename) - no reliable explicit numbering found"


def main():
    parser = argparse.ArgumentParser(description="Merge structured utterances into embeddable chunks.")
    parser.add_argument("--input-dir", default="transcripts/structured", help="Directory of Phase 1 structured transcripts")
    parser.add_argument("--output-dir", default="transcripts", help="Directory to write chunks/ and episode_manifest.json into")
    parser.add_argument("--force", action="store_true", help="Re-chunk even if chunk output already exists")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"ERROR: input dir {input_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    structured_files = sorted(input_dir.glob("*.json"))
    if not structured_files:
        print(f"No structured transcripts found in {input_dir}.")
        sys.exit(0)

    source_files = []
    episode_data = {}
    for path in structured_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        episode_data[data["episode_id"]] = data
        source_files.append(data["source_file"])

    episode_numbers, method_note = derive_episode_number(source_files)
    print(f"Episode numbering method: {method_note}\n")

    manifest = []
    failures = []

    for path in structured_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        episode_id = data["episode_id"]
        source_file = data["source_file"]
        duration_seconds = data.get("duration_seconds")
        chunk_path = chunks_dir / f"{episode_id}.json"

        if chunk_path.exists() and not args.force:
            print(f"[{episode_id}] chunks already exist, skipping (use --force to re-run)")
            with open(chunk_path, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
            manifest.append({
                "episode_number": episode_numbers[source_file],
                "episode_id": episode_id,
                "source_file": source_file,
                "duration_seconds": duration_seconds,
                "chunk_count": len(existing_chunks),
            })
            continue

        try:
            groups = group_utterances(data["utterances"])
            chunks = [build_chunk(episode_id, g, i + 1) for i, g in enumerate(groups)]
            warnings = validate_chunks(episode_id, chunks, duration_seconds)
            for w in warnings:
                print(f"    warning: {w}")

            with open(chunk_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)

            print(f"[{episode_id}] {len(data['utterances'])} utterances -> {len(chunks)} chunks")

            manifest.append({
                "episode_number": episode_numbers[source_file],
                "episode_id": episode_id,
                "source_file": source_file,
                "duration_seconds": duration_seconds,
                "chunk_count": len(chunks),
            })
        except Exception as exc:
            print(f"[{episode_id}] FAILED: {exc}", file=sys.stderr)
            failures.append((episode_id, str(exc)))

    manifest.sort(key=lambda e: e["episode_number"])
    manifest_path = output_dir / "episode_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote manifest for {len(manifest)} episode(s) to {manifest_path}")

    if failures:
        print("\n=== Failures ===")
        for episode_id, err in failures:
            print(f"{episode_id}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
