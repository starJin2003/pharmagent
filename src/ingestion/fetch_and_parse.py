"""Fetch drug labels from openFDA and extract relevant sections."""

import asyncio
import json
import logging

from src.api_clients.openfda_client import OpenFDAClient
from src.config import (
    API_DELAY,
    LABELS_PER_DRUG,
    PROCESSED_DIR,
    SECTIONS_TO_EXTRACT,
    TOP_DRUGS_LIMIT,
)

logger = logging.getLogger(__name__)


def _extract_sections(label: dict, drug_name: str) -> list[dict]:
    """Extract configured sections from a single label into document dicts.

    Args:
        label: Raw label dict from the openFDA API response.
        drug_name: Lowercase generic drug name.

    Returns:
        List of document dicts with text and metadata.
    """
    openfda = label.get("openfda", {})
    brand_names = openfda.get("brand_name", [])
    route = openfda.get("route", ["UNKNOWN"])[0] if openfda.get("route") else "UNKNOWN"
    label_id = label.get("id", f"{drug_name}_{route}")

    documents: list[dict] = []
    for section in SECTIONS_TO_EXTRACT:
        texts = label.get(section, [])
        if not texts:
            continue
        text = " ".join(texts).strip()
        if not text:
            continue
        documents.append({
            "text": text,
            "metadata": {
                "drug_generic_name": drug_name.lower(),
                "drug_brand_names": brand_names,
                "section_type": section,
                "source": "openFDA",
                "route": route,
                "label_id": label_id,
            },
        })
    return documents


async def fetch_and_parse() -> list[dict]:
    """Fetch top drug labels from openFDA, deduplicate, and extract sections.

    Returns:
        List of document dicts with text and metadata, saved to sections.json.
    """
    client = OpenFDAClient()

    # Step 1: Get top drug names
    logger.info("Fetching top %d drug names...", TOP_DRUGS_LIMIT)
    drug_names = await client.get_top_drugs(limit=TOP_DRUGS_LIMIT)
    if not drug_names:
        logger.error("No drug names returned from openFDA count endpoint.")
        return []
    logger.info("Got %d drug names.", len(drug_names))

    # Step 2: Fetch labels for each drug with deduplication
    seen: set[str] = set()  # generic_name + route keys
    all_documents: list[dict] = []

    for i, drug_name in enumerate(drug_names):
        if i > 0:
            await asyncio.sleep(API_DELAY)

        labels = await client.search_labels(drug_name, limit=LABELS_PER_DRUG)
        if not labels:
            continue

        for label in labels:
            openfda = label.get("openfda", {})
            generic_names = openfda.get("generic_name", [drug_name])
            route_list = openfda.get("route", ["UNKNOWN"])
            generic = generic_names[0].lower() if generic_names else drug_name.lower()
            route = route_list[0] if route_list else "UNKNOWN"

            dedup_key = f"{generic}|{route}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            docs = _extract_sections(label, generic)
            all_documents.extend(docs)

        if (i + 1) % 50 == 0:
            logger.info(
                "Processed %d/%d drugs, %d documents so far.",
                i + 1,
                len(drug_names),
                len(all_documents),
            )

    logger.info(
        "Extraction complete: %d documents from %d unique drug/route combos.",
        len(all_documents),
        len(seen),
    )

    # Step 3: Save to processed directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "sections.json"
    with open(output_path, "w") as f:
        json.dump(all_documents, f, indent=2)
    logger.info("Saved sections to %s", output_path)

    return all_documents
