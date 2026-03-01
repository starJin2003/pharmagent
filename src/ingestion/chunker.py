"""Section-aware chunking of FDA drug label documents."""

import json
import logging
import re

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> float:
    """Approximate token count without tiktoken.

    Args:
        text: Input text string.

    Returns:
        Estimated token count.
    """
    return len(text.split()) * 1.3


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines or single newlines.

    Args:
        text: Input text.

    Returns:
        List of non-empty paragraph strings.
    """
    parts = re.split(r"\n\s*\n|\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Args:
        text: Input text.

    Returns:
        List of non-empty sentence strings.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _group_sentences(sentences: list[str], min_tokens: float, max_tokens: float) -> list[str]:
    """Group sentences into chunks within a target token range.

    Args:
        sentences: List of sentence strings.
        min_tokens: Minimum target tokens per chunk.
        max_tokens: Maximum target tokens per chunk.

    Returns:
        List of grouped text chunks.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0.0

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)
        if current and current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sent_tokens
        else:
            current.append(sentence)
            current_tokens += sent_tokens

    if current:
        # Merge short trailing chunk into previous if possible
        trailing = " ".join(current)
        if chunks and _estimate_tokens(trailing) < min_tokens:
            chunks[-1] = chunks[-1] + " " + trailing
        else:
            chunks.append(trailing)

    return chunks


def _chunk_drug_interactions(text: str) -> list[str]:
    """Chunk drug_interactions: paragraph split, sentence groups if paragraph > 500 tokens.

    Args:
        text: Full section text.

    Returns:
        List of chunk strings.
    """
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    for para in paragraphs:
        if _estimate_tokens(para) > 500:
            sentences = _split_sentences(para)
            chunks.extend(_group_sentences(sentences, 150, 400))
        else:
            chunks.append(para)
    return chunks


def _chunk_warnings(text: str) -> list[str]:
    """Chunk warnings/warnings_and_cautions/boxed_warning: keep whole if <600, else paragraph.

    Args:
        text: Full section text.

    Returns:
        List of chunk strings.
    """
    if _estimate_tokens(text) < 600:
        return [text]
    return _split_paragraphs(text)


def _chunk_adverse_reactions(text: str) -> list[str]:
    """Chunk adverse_reactions: split by paragraph (200-500 tokens target).

    Args:
        text: Full section text.

    Returns:
        List of chunk strings.
    """
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    for para in paragraphs:
        if _estimate_tokens(para) > 500:
            sentences = _split_sentences(para)
            chunks.extend(_group_sentences(sentences, 200, 500))
        else:
            chunks.append(para)
    return chunks


def _chunk_contraindications(text: str) -> list[str]:
    """Chunk contraindications: keep whole.

    Args:
        text: Full section text.

    Returns:
        List containing the full text as a single chunk.
    """
    return [text]


def _chunk_indications(text: str) -> list[str]:
    """Chunk indications_and_usage: keep whole if <500 tokens, else paragraph split.

    Args:
        text: Full section text.

    Returns:
        List of chunk strings.
    """
    if _estimate_tokens(text) < 500:
        return [text]
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    for para in paragraphs:
        if _estimate_tokens(para) > 500:
            sentences = _split_sentences(para)
            chunks.extend(_group_sentences(sentences, 200, 500))
        else:
            chunks.append(para)
    return chunks


_CHUNKERS = {
    "drug_interactions": _chunk_drug_interactions,
    "warnings": _chunk_warnings,
    "warnings_and_cautions": _chunk_warnings,
    "boxed_warning": _chunk_warnings,
    "adverse_reactions": _chunk_adverse_reactions,
    "contraindications": _chunk_contraindications,
    "indications_and_usage": _chunk_indications,
}


def chunk_sections(sections: list[dict]) -> list[dict]:
    """Chunk extracted sections into smaller pieces following section-specific strategies.

    Args:
        sections: List of document dicts from fetch_and_parse, each with text and metadata.

    Returns:
        List of chunk dicts with text and metadata (including chunk_index).
    """
    all_chunks: list[dict] = []
    # Track chunk_index per drug
    drug_chunk_counts: dict[str, int] = {}

    for doc in sections:
        text = doc["text"]
        metadata = doc["metadata"]
        section_type = metadata["section_type"]
        drug_name = metadata["drug_generic_name"]

        chunker = _CHUNKERS.get(section_type)
        if chunker is None:
            logger.warning("No chunker for section type: %s", section_type)
            continue

        text_chunks = chunker(text)

        for chunk_text in text_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            idx = drug_chunk_counts.get(drug_name, 0)
            drug_chunk_counts[drug_name] = idx + 1

            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "drug_generic_name": metadata["drug_generic_name"],
                    "drug_brand_names": metadata["drug_brand_names"],
                    "section_type": section_type,
                    "source": metadata["source"],
                    "route": metadata["route"],
                    "label_id": metadata["label_id"],
                    "chunk_index": idx,
                },
            })

    logger.info("Chunking complete: %d chunks from %d sections.", len(all_chunks), len(sections))

    # Save to processed directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "chunks.json"
    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)
    logger.info("Saved chunks to %s", output_path)

    return all_chunks
