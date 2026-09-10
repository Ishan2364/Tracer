"""Embed a query and run a similarity search over the chunk index - unscoped (Phase 3)
or scoped to a single episode (Phase 4)."""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions
from langsmith import traceable

import config

_collection = None

# Incremented on every actual Chroma `collection.query()` call. Phase 4's routing
# logic is validated partly by call count (e.g. a follow-up must trigger zero new
# queries), so this is a cheap, explicit hook for that rather than mocking the client.
QUERY_CALL_COUNT = 0


def get_embedding_function():
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
        ef(["sanity check"])
        return ef
    except Exception as exc:
        print(f"Primary embedding model '{config.EMBEDDING_MODEL}' failed to load ({exc}); "
              f"falling back to '{config.EMBEDDING_MODEL_FALLBACK}'")
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_FALLBACK)


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.INDEX_DIR)
        _collection = client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    return _collection


def _chunks_from_result(result) -> list[dict]:
    if not result["ids"][0]:
        return []
    chunks = []
    for cid, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append({
            "chunk_id": cid,
            "text": doc,
            "similarity": 1 - dist,
            "episode_id": meta["episode_id"],
            "episode_number": meta["episode_number"],
            "start": meta["start"],
            "end": meta["end"],
            "speakers": meta["speakers"],
            "utterance_ids": meta["utterance_ids"],
        })
    return chunks


@traceable(run_type="retriever", name="retrieve_chunks")
def retrieve(query: str, n_results: int | None = None) -> list[dict]:
    """Unscoped similarity search across the full collection."""
    global QUERY_CALL_COUNT
    n_results = n_results or config.N_RESULTS
    collection = get_collection()
    result = collection.query(query_texts=[query], n_results=n_results)
    QUERY_CALL_COUNT += 1
    return _chunks_from_result(result)


@traceable(run_type="retriever", name="retrieve_chunks_scoped")
def retrieve_scoped(query: str, episode_id: str, n_results: int | None = None) -> list[dict]:
    """Similarity search restricted to a single episode via a `where` filter."""
    global QUERY_CALL_COUNT
    n_results = n_results or config.N_RESULTS
    collection = get_collection()
    result = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"episode_id": episode_id},
    )
    QUERY_CALL_COUNT += 1
    return _chunks_from_result(result)
