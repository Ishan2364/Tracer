"""Phase 2 (step 2): embed chunks locally and load them into a persistent Chroma collection.

Usage:
    python src/build_index.py [--input-dir transcripts/chunks] [--index-dir index/chroma_db] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

import config

COLLECTION_NAME = config.COLLECTION_NAME
PRIMARY_MODEL = config.EMBEDDING_MODEL
FALLBACK_MODEL = config.EMBEDDING_MODEL_FALLBACK
BATCH_SIZE = 100


def get_embedding_function():
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=PRIMARY_MODEL)
        ef(["sanity check"])
        print(f"Using embedding model: {PRIMARY_MODEL}")
        return ef
    except Exception as exc:
        print(f"Primary embedding model '{PRIMARY_MODEL}' failed to load ({exc}); "
              f"falling back to '{FALLBACK_MODEL}'")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=FALLBACK_MODEL)
        print(f"Using embedding model: {FALLBACK_MODEL}")
        return ef


def load_chunks(chunks_dir: Path) -> list[dict]:
    all_chunks = []
    for path in sorted(chunks_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            all_chunks.extend(json.load(f))
    return all_chunks


def load_episode_numbers(manifest_path: Path) -> dict[str, int]:
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {e["episode_id"]: e["episode_number"] for e in manifest}


def main():
    parser = argparse.ArgumentParser(description="Embed chunks and load them into a persistent Chroma collection.")
    parser.add_argument("--input-dir", default=config.CHUNKS_DIR, help="Directory of Phase 2 chunk files")
    parser.add_argument("--manifest", default=config.MANIFEST_PATH, help="Path to episode manifest")
    parser.add_argument("--index-dir", default=config.INDEX_DIR, help="Directory for the persistent Chroma index")
    parser.add_argument("--force", action="store_true", help="Wipe and rebuild the collection from scratch")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input dir {input_dir} does not exist. Run chunk_transcripts.py first.", file=sys.stderr)
        sys.exit(1)

    all_chunks = load_chunks(input_dir)
    if not all_chunks:
        print(f"No chunks found in {input_dir}.")
        sys.exit(0)

    episode_numbers = load_episode_numbers(Path(args.manifest))

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir))

    if args.force:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}' (--force).")
        except Exception:
            pass

    embedding_function = get_embedding_function()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    all_ids = [c["chunk_id"] for c in all_chunks]
    existing_ids: set[str] = set()
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        found = collection.get(ids=batch_ids)
        existing_ids.update(found["ids"])

    new_chunks = [c for c in all_chunks if c["chunk_id"] not in existing_ids]
    print(f"{len(all_chunks)} total chunks, {len(existing_ids)} already indexed, {len(new_chunks)} new.")

    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i:i + BATCH_SIZE]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "episode_id": c["episode_id"],
                    "episode_number": episode_numbers.get(c["episode_id"], -1),
                    "start": c["start"],
                    "end": c["end"],
                    "speakers": c["speakers"],
                    "utterance_ids": c["utterance_ids"],
                }
                for c in batch
            ],
        )
        print(f"  added batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    actual_count = collection.count()
    print(f"\nCollection '{COLLECTION_NAME}' now has {actual_count} entries (expected {len(all_chunks)}).")
    if actual_count != len(all_chunks):
        print("ERROR: collection count does not match total chunk count.", file=sys.stderr)
        sys.exit(1)

    print("\n=== Sanity query (unscoped): 'physics', n_results=3 ===")
    result = collection.query(query_texts=["physics"], n_results=3)
    for cid, doc, meta in zip(result["ids"][0], result["documents"][0], result["metadatas"][0]):
        print(f"  {cid} | episode={meta.get('episode_id')} | [{meta.get('start'):.1f}-{meta.get('end'):.1f}] | {doc[:80]!r}")

    sample_episode_id = all_chunks[0]["episode_id"]
    print(f"\n=== Sanity query (scoped to episode_id={sample_episode_id!r}): 'physics', n_results=3 ===")
    scoped_result = collection.query(
        query_texts=["physics"],
        n_results=3,
        where={"episode_id": sample_episode_id},
    )
    all_match = True
    for cid, doc, meta in zip(scoped_result["ids"][0], scoped_result["documents"][0], scoped_result["metadatas"][0]):
        matches = meta.get("episode_id") == sample_episode_id
        all_match = all_match and matches
        print(f"  {cid} | episode={meta.get('episode_id')} | match={matches} | [{meta.get('start'):.1f}-{meta.get('end'):.1f}]")

    if not all_match:
        print("ERROR: scoped query returned results outside the requested episode_id.", file=sys.stderr)
        sys.exit(1)

    print("\nIndex build complete and verified.")


if __name__ == "__main__":
    main()
