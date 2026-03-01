"""Orchestrate the full data ingestion pipeline."""

import asyncio
import json
import logging

from src.config import PROCESSED_DIR
from src.ingestion.chunker import chunk_sections
from src.ingestion.fetch_and_parse import fetch_and_parse
from src.ingestion.indexer import build_indexes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the full ingestion pipeline: fetch → chunk → index."""
    logger.info("=== Starting ingestion pipeline ===")

    sections_path = PROCESSED_DIR / "sections.json"
    chunks_path = PROCESSED_DIR / "chunks.json"

    # Step 1: Fetch and parse labels (skip if sections.json exists)
    if sections_path.exists():
        logger.info("Step 1/3: Loading cached sections from %s", sections_path)
        with open(sections_path) as f:
            sections = json.load(f)
    else:
        logger.info("Step 1/3: Fetching and parsing drug labels...")
        sections = await fetch_and_parse()
    logger.info("Step 1 complete: %d sections.", len(sections))
    if not sections:
        logger.error("No sections extracted. Aborting.")
        return

    # Step 2: Chunk sections (skip if chunks.json exists)
    if chunks_path.exists():
        logger.info("Step 2/3: Loading cached chunks from %s", chunks_path)
        with open(chunks_path) as f:
            chunks = json.load(f)
    else:
        logger.info("Step 2/3: Chunking sections...")
        chunks = chunk_sections(sections)
    logger.info("Step 2 complete: %d chunks.", len(chunks))
    if not chunks:
        logger.error("No chunks created. Aborting.")
        return

    # Step 3: Build indexes
    logger.info("Step 3/3: Building indexes...")
    build_indexes(chunks)
    logger.info("Step 3 complete.")

    logger.info("=== Ingestion pipeline finished ===")


if __name__ == "__main__":
    asyncio.run(main())
