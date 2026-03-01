"""Evaluation pipeline for PharmAgent: Recall@5, MRR, Faithfulness, Citation Accuracy."""

import asyncio
import json
import logging
import re
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.agents.graph import graph
from src.config import DATA_DIR, LLM_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
_OUTPUT_PATH = DATA_DIR / "eval_results.json"

_llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)


def _load_golden_dataset() -> list[dict]:
    """Load the golden evaluation dataset.

    Returns:
        List of evaluation entries.
    """
    with open(_GOLDEN_PATH) as f:
        return json.load(f)


def _chunk_matches_section(chunk: dict, section: dict) -> bool:
    """Check if a retrieved chunk matches an expected relevant section.

    Args:
        chunk: Retrieved chunk with metadata.
        section: Expected section dict with drug_generic_name and section_type.

    Returns:
        True if the chunk's drug and section match.
    """
    meta = chunk.get("metadata", {})
    chunk_drug = meta.get("drug_generic_name", "").lower()
    chunk_section = meta.get("section_type", "")
    expected_drug = section["drug_generic_name"].lower()
    expected_section = section["section_type"]

    drug_match = expected_drug in chunk_drug or chunk_drug.startswith(expected_drug)
    section_match = chunk_section == expected_section

    return drug_match and section_match


def compute_recall_at_5(chunks: list[dict], relevant_sections: list[dict]) -> float:
    """Compute Recall@5: did any top-5 chunk match a relevant section?

    Args:
        chunks: Top-5 retrieved chunks.
        relevant_sections: Expected relevant sections from golden dataset.

    Returns:
        1.0 if any chunk matches any relevant section, else 0.0.
    """
    if not relevant_sections:
        return 1.0

    for chunk in chunks[:5]:
        for section in relevant_sections:
            if _chunk_matches_section(chunk, section):
                return 1.0
    return 0.0


def compute_mrr(chunks: list[dict], relevant_sections: list[dict]) -> float:
    """Compute MRR: 1/rank of the first relevant chunk.

    Args:
        chunks: Top-5 retrieved chunks.
        relevant_sections: Expected relevant sections from golden dataset.

    Returns:
        1/rank of first matching chunk, or 0.0 if none match.
    """
    if not relevant_sections:
        return 1.0

    for rank, chunk in enumerate(chunks[:5], 1):
        for section in relevant_sections:
            if _chunk_matches_section(chunk, section):
                return 1.0 / rank
    return 0.0


async def compute_faithfulness(answer: str, chunks: list[dict]) -> float:
    """Compute faithfulness: fraction of answer sentences supported by sources.

    Uses GPT-4o-mini to judge each sentence as SUPPORTED, NOT_SUPPORTED, or DISCLAIMER.

    Args:
        answer: The generated answer text.
        chunks: Retrieved chunks used as context.

    Returns:
        SUPPORTED / (SUPPORTED + NOT_SUPPORTED), or 1.0 if no judgeable sentences.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 1.0

    context = "\n\n".join(c["text"] for c in chunks)

    prompt = (
        "You are an evaluation judge. For each numbered sentence below, determine if it is:\n"
        "- SUPPORTED: the claim is backed by the provided context\n"
        "- NOT_SUPPORTED: the claim makes a factual assertion not found in the context\n"
        "- DISCLAIMER: the sentence is a safety disclaimer, hedge, or recommendation to consult a doctor\n\n"
        "Important rules:\n"
        "- If the answer states that no interaction or information was found in the sources, "
        "and the provided context indeed does not mention such interaction, mark as SUPPORTED "
        "— this accurately reflects the source content.\n"
        "- Hedging language like 'no specific interaction was found' or 'the sources do not indicate' "
        "is SUPPORTED when it correctly reflects what the sources contain.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Sentences:\n"
    )
    for i, sent in enumerate(sentences, 1):
        prompt += f"{i}. {sent}\n"

    prompt += (
        "\nRespond with ONLY one label per line in the format: "
        '"N. LABEL" (e.g., "1. SUPPORTED"). No explanations.'
    )

    try:
        response = await _llm.ainvoke(prompt)
        raw = response.content.strip()

        supported = 0
        not_supported = 0
        for line in raw.splitlines():
            line = line.strip().upper()
            if "SUPPORTED" in line and "NOT_SUPPORTED" not in line:
                supported += 1
            elif "NOT_SUPPORTED" in line:
                not_supported += 1

        total = supported + not_supported
        if total == 0:
            return 1.0
        return supported / total
    except Exception as exc:
        logger.error("Faithfulness evaluation failed: %s", exc)
        return 0.0


def compute_citation_accuracy(answer: str, chunks: list[dict]) -> float:
    """Compute citation accuracy: fraction of factual claims with valid [Source N] refs.

    Args:
        answer: The generated answer text.
        chunks: Retrieved chunks (to validate source references).

    Returns:
        Fraction of sentences containing factual claims that have valid citations.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 1.0

    max_source = len(chunks)
    factual_count = 0
    cited_count = 0

    disclaimer_patterns = re.compile(
        r"(consult|disclaimer|educational purposes|healthcare provider|pharmacist|"
        r"could not find|try rephrasing|insufficient)",
        re.IGNORECASE,
    )

    for sent in sentences:
        if disclaimer_patterns.search(sent):
            continue

        # Consider it a factual claim if it's not a question or pure transition
        if len(sent.split()) < 4:
            continue

        factual_count += 1
        refs = re.findall(r"\[Source (\d+)\]", sent)
        valid_refs = [r for r in refs if 1 <= int(r) <= max_source]
        if valid_refs:
            cited_count += 1

    if factual_count == 0:
        return 1.0
    return cited_count / factual_count


async def _run_single_eval(entry: dict) -> dict:
    """Run evaluation for a single golden dataset entry.

    Args:
        entry: A golden dataset entry.

    Returns:
        Dict with entry id, metrics, and details.
    """
    query = entry["query"]
    category = entry["category"]

    initial_state = {
        "original_query": query,
        "safety_flag": "",
        "safety_message": None,
        "resolved_drugs": [],
        "query_type": "",
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "sources_text": "",
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Graph invocation failed for %s: %s", entry["id"], exc)
        return {
            "id": entry["id"],
            "query": query,
            "category": category,
            "error": str(exc),
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "faithfulness": 0.0,
            "citation_accuracy": 0.0,
        }

    # For out-of-scope queries, check that safety guard caught them
    if category == "out_of_scope":
        caught = result["safety_flag"] != "ok"
        return {
            "id": entry["id"],
            "query": query,
            "category": category,
            "safety_flag": result["safety_flag"],
            "caught_out_of_scope": caught,
            "recall_at_5": 1.0 if caught else 0.0,
            "mrr": 1.0 if caught else 0.0,
            "faithfulness": 1.0 if caught else 0.0,
            "citation_accuracy": 1.0 if caught else 0.0,
        }

    chunks = result.get("retrieved_chunks", [])
    answer = result.get("answer", "")
    relevant_sections = entry.get("relevant_sections", [])

    recall = compute_recall_at_5(chunks, relevant_sections)
    mrr = compute_mrr(chunks, relevant_sections)
    faithfulness = await compute_faithfulness(answer, chunks)
    citation_acc = compute_citation_accuracy(answer, chunks)

    return {
        "id": entry["id"],
        "query": query,
        "category": category,
        "safety_flag": result.get("safety_flag", ""),
        "resolved_drugs": [d.get("generic_name", "") for d in result.get("resolved_drugs", [])],
        "query_type": result.get("query_type", ""),
        "num_chunks": len(chunks),
        "answer_length": len(answer),
        "recall_at_5": recall,
        "mrr": mrr,
        "faithfulness": faithfulness,
        "citation_accuracy": citation_acc,
    }


async def main() -> None:
    """Run the full evaluation pipeline and save results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset = _load_golden_dataset()
    logger.info("Loaded %d evaluation entries.", len(dataset))

    results: list[dict] = []
    for i, entry in enumerate(dataset):
        logger.info("[%d/%d] Evaluating: %s", i + 1, len(dataset), entry["query"][:60])
        result = await _run_single_eval(entry)
        results.append(result)
        logger.info(
            "  R@5=%.2f  MRR=%.2f  Faith=%.2f  Cite=%.2f",
            result["recall_at_5"],
            result["mrr"],
            result["faithfulness"],
            result["citation_accuracy"],
        )

    # Compute aggregate metrics
    n = len(results)
    agg_recall = sum(r["recall_at_5"] for r in results) / n if n else 0
    agg_mrr = sum(r["mrr"] for r in results) / n if n else 0
    agg_faith = sum(r["faithfulness"] for r in results) / n if n else 0
    agg_cite = sum(r["citation_accuracy"] for r in results) / n if n else 0

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    category_metrics: dict[str, dict] = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cn = len(cat_results)
        category_metrics[cat] = {
            "count": cn,
            "recall_at_5": sum(r["recall_at_5"] for r in cat_results) / cn,
            "mrr": sum(r["mrr"] for r in cat_results) / cn,
            "faithfulness": sum(r["faithfulness"] for r in cat_results) / cn,
            "citation_accuracy": sum(r["citation_accuracy"] for r in cat_results) / cn,
        }

    output = {
        "aggregate": {
            "num_entries": n,
            "recall_at_5": round(agg_recall, 4),
            "mrr": round(agg_mrr, 4),
            "faithfulness": round(agg_faith, 4),
            "citation_accuracy": round(agg_cite, 4),
        },
        "by_category": {
            cat: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
            for cat, m in category_metrics.items()
        },
        "per_query": results,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("  Entries:            %d", n)
    logger.info("  Recall@5:           %.4f (target >= 0.75)", agg_recall)
    logger.info("  MRR:                %.4f (target >= 0.50)", agg_mrr)
    logger.info("  Faithfulness:       %.4f (target >= 0.85)", agg_faith)
    logger.info("  Citation Accuracy:  %.4f (target >= 0.80)", agg_cite)
    logger.info("-" * 60)
    for cat, m in category_metrics.items():
        logger.info(
            "  %-22s (n=%d) R@5=%.2f MRR=%.2f Faith=%.2f Cite=%.2f",
            cat, m["count"], m["recall_at_5"], m["mrr"],
            m["faithfulness"], m["citation_accuracy"],
        )
    logger.info("Results saved to %s", _OUTPUT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
