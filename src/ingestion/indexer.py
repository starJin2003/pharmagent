"""Embed chunks and build FAISS + BM25 indexes."""

import json
import logging
import pickle
import time

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL, INDEX_DIR, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50  # smaller batches to stay under TPM rate limit
_MAX_WORDS = 4000  # conservative limit — medical text tokenizes at ~1.7-2x words
_BATCH_DELAY = 1.0  # seconds between batches to avoid TPM rate limit


def _truncate_for_embedding(text: str) -> str:
    """Truncate text to fit within the embedding model's token limit.

    Args:
        text: Input text string.

    Returns:
        Truncated text if it exceeded the limit, otherwise unchanged.
    """
    words = text.split()
    if len(words) > _MAX_WORDS:
        logger.warning("Truncating chunk from %d to %d words for embedding.", len(words), _MAX_WORDS)
        return " ".join(words[:_MAX_WORDS])
    return text


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts via OpenAI in batches with rate limit handling.

    Args:
        texts: List of text strings to embed.

    Returns:
        NumPy array of shape (len(texts), EMBEDDING_DIM).
    """
    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=5)
    all_embeddings: list[list[float]] = []
    safe_texts = [_truncate_for_embedding(t) for t in texts]
    total_batches = (len(safe_texts) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for i in range(0, len(safe_texts), _BATCH_SIZE):
        batch = safe_texts[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1
        logger.info("Embedding batch %d/%d (%d texts)...", batch_num, total_batches, len(batch))

        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # Delay between batches to stay under TPM rate limit
        if batch_num < total_batches:
            time.sleep(_BATCH_DELAY)

    return np.array(all_embeddings, dtype=np.float32)


def build_indexes(chunks: list[dict]) -> None:
    """Build FAISS and BM25 indexes from chunks and save all artifacts.

    Args:
        chunks: List of chunk dicts, each with "text" and "metadata" keys.
    """
    if not chunks:
        logger.error("No chunks provided, skipping index build.")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    texts = [chunk["text"] for chunk in chunks]

    # Step 1: Embed all texts
    logger.info("Embedding %d chunks...", len(texts))
    embeddings = _embed_texts(texts)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # Step 2: Save embeddings
    embeddings_path = INDEX_DIR / "embeddings.npy"
    np.save(str(embeddings_path), embeddings)
    logger.info("Saved embeddings to %s", embeddings_path)

    # Step 3: Save chunk store
    chunk_store_path = INDEX_DIR / "chunk_store.json"
    with open(chunk_store_path, "w") as f:
        json.dump(chunks, f, indent=2)
    logger.info("Saved chunk store to %s", chunk_store_path)

    # Step 4: Build FAISS index (L2 normalize → IndexFlatIP for cosine similarity)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    faiss_path = INDEX_DIR / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    logger.info("Saved FAISS index (%d vectors) to %s", index.ntotal, faiss_path)

    # Step 5: Build BM25 index
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_path = INDEX_DIR / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info("Saved BM25 index to %s", bm25_path)

    logger.info("Index build complete: %d chunks indexed.", len(chunks))
