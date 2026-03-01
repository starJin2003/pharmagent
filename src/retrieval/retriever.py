"""FAISS + BM25 hybrid retrieval with RRF fusion."""

import json
import logging
import pickle

import faiss
import numpy as np
from openai import AsyncOpenAI

from src.config import EMBEDDING_MODEL, INDEX_DIR, OPENAI_API_KEY

logger = logging.getLogger(__name__)

# Module-level singletons loaded once at import time
_faiss_index: faiss.IndexFlatIP | None = None
_bm25 = None
_chunk_store: list[dict] = []
_embedding_client: AsyncOpenAI | None = None


def _load_resources() -> None:
    """Load FAISS index, BM25 object, chunk store, and embedding client."""
    global _faiss_index, _bm25, _chunk_store, _embedding_client

    if _faiss_index is not None:
        return

    logger.info("Loading retrieval indexes...")

    faiss_path = INDEX_DIR / "faiss.index"
    _faiss_index = faiss.read_index(str(faiss_path))
    logger.info("Loaded FAISS index: %d vectors.", _faiss_index.ntotal)

    bm25_path = INDEX_DIR / "bm25.pkl"
    with open(bm25_path, "rb") as f:
        _bm25 = pickle.load(f)
    logger.info("Loaded BM25 index.")

    chunk_store_path = INDEX_DIR / "chunk_store.json"
    with open(chunk_store_path) as f:
        _chunk_store = json.load(f)
    logger.info("Loaded chunk store: %d chunks.", len(_chunk_store))

    _embedding_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("Retrieval resources loaded.")


async def _embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Args:
        query: The search query text.

    Returns:
        Normalized embedding vector of shape (1, EMBEDDING_DIM).
    """
    _load_resources()
    response = await _embedding_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[query]
    )
    vec = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


def _rrf_fuse(
    faiss_indices: list[int],
    bm25_indices: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of two ranked lists.

    Args:
        faiss_indices: Ranked document indices from FAISS.
        bm25_indices: Ranked document indices from BM25.
        k: RRF constant (default 60).

    Returns:
        List of (doc_index, rrf_score) tuples sorted by score descending.
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(faiss_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(bm25_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def retrieve(
    query: str, drug_names: list[str], top_k: int = 5
) -> list[dict]:
    """Retrieve top-k chunks using hybrid FAISS + BM25 search with RRF fusion.

    Args:
        query: The user's search query.
        drug_names: List of resolved generic drug names to enrich the query.
        top_k: Number of results to return.

    Returns:
        List of chunk dicts with text, metadata, and score.
    """
    _load_resources()

    # Enrich query with drug names
    enriched_query = " ".join(drug_names) + " " + query
    logger.info("Retrieval query: %s", enriched_query[:120])

    # FAISS search
    query_vec = await _embed_query(enriched_query)
    _, faiss_ids = _faiss_index.search(query_vec, 20)
    faiss_indices = [int(idx) for idx in faiss_ids[0] if idx >= 0]

    # BM25 search
    tokenized_query = enriched_query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(-bm25_scores)[:20]
    bm25_indices = [int(idx) for idx in bm25_top]

    # RRF fusion
    fused = _rrf_fuse(faiss_indices, bm25_indices)

    # Build result list
    results: list[dict] = []
    for idx, score in fused[:top_k]:
        if idx < 0 or idx >= len(_chunk_store):
            continue
        chunk = _chunk_store[idx]
        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": score,
        })

    logger.info("Retrieved %d chunks.", len(results))
    return results
