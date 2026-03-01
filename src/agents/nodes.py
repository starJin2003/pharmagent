"""Node functions for the PharmAgent LangGraph workflow."""

import json
import logging
import re

from langchain_openai import ChatOpenAI

from src.agents.state import AgentState
from src.api_clients.rxnorm_client import RxNormClient
from src.config import LLM_MODEL, OPENAI_API_KEY
from src.retrieval import retriever

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)
_rxnorm = RxNormClient()

# --- Keyword sets for input safety check ---

_EMERGENCY_KEYWORDS = re.compile(
    r"\b(overdose|suicide|suicidal|kill myself|want to die|self[- ]?harm|"
    r"poisoning|swallowed too many|took too many|can't breathe|"
    r"allergic reaction|anaphylaxis|heart attack|seizure|stroke|"
    r"unconscious|not breathing|choking)\b",
    re.IGNORECASE,
)

_NEEDS_DOCTOR_KEYWORDS = re.compile(
    r"\b(diagnose|diagnosis|what do i have|prescribe|prescription|"
    r"how much should i take|what dose|dosage for me|"
    r"should i stop taking|replace my medication|"
    r"am i sick|what is wrong with me)\b",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_KEYWORDS = re.compile(
    r"\b(recipe|weather|stock|crypto|bitcoin|politics|"
    r"who is the president|capital of|math problem|"
    r"write me a|translate|code|programming)\b",
    re.IGNORECASE,
)

_DISCLAIMER = (
    "\n\n\u26a0\ufe0f This information is for educational purposes only. "
    "Always consult your healthcare provider or pharmacist."
)

_DOSAGE_PATTERN = re.compile(
    r"\b(take|use)\s+\d+\s*(mg|ml|mcg|g|tablet|capsule|pill)s?\b",
    re.IGNORECASE,
)


async def input_safety_check(state: AgentState) -> AgentState:
    """Check the user's query for safety concerns using regex/keyword matching.

    Args:
        state: Current agent state.

    Returns:
        Updated state with safety_flag and optional safety_message set.
    """
    query = state["original_query"]

    if _EMERGENCY_KEYWORDS.search(query):
        state["safety_flag"] = "emergency"
        state["safety_message"] = (
            "If you are experiencing a medical emergency, please call 911 "
            "(or your local emergency number) immediately or go to the "
            "nearest emergency room. Do not rely on this tool for "
            "emergency medical advice."
        )
        return state

    if _NEEDS_DOCTOR_KEYWORDS.search(query):
        state["safety_flag"] = "needs_doctor"
        state["safety_message"] = (
            "This question involves personal medical decisions such as "
            "diagnosis or dosage guidance. Please consult your healthcare "
            "provider or pharmacist for advice tailored to your situation."
            + _DISCLAIMER
        )
        return state

    if _OUT_OF_SCOPE_KEYWORDS.search(query):
        state["safety_flag"] = "out_of_scope"
        state["safety_message"] = (
            "I can only answer questions about drug interactions, side effects, "
            "contraindications, and general drug information based on FDA data. "
            "Your question appears to be outside this scope."
        )
        return state

    state["safety_flag"] = "ok"
    return state


async def resolve_drugs(state: AgentState) -> AgentState:
    """Extract drug names from the query via LLM, then resolve each with RxNorm.

    Args:
        state: Current agent state with original_query.

    Returns:
        Updated state with resolved_drugs populated.
    """
    query = state["original_query"]

    try:
        extraction_prompt = (
            "Extract all drug names (brand or generic) from the following question. "
            "Return ONLY a JSON array of strings, nothing else.\n\n"
            f"Question: {query}\n\n"
            "Example output: [\"ibuprofen\", \"warfarin\"]"
        )
        response = await _llm.ainvoke(extraction_prompt)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        drug_names: list[str] = json.loads(raw)
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        logger.error("Drug name extraction failed: %s", exc)
        state["resolved_drugs"] = []
        return state

    resolved: list[dict] = []
    for name in drug_names:
        try:
            result = await _rxnorm.resolve_drug_name(name)
            if result:
                resolved.append({
                    "input_name": name,
                    "generic_name": result["generic_name"],
                    "brand_names": result.get("brand_names", []),
                    "rxcui": result["rxcui"],
                })
            else:
                logger.error("RxNorm could not resolve: %s", name)
        except Exception as exc:
            logger.error("RxNorm resolution error for %s: %s", name, exc)

    state["resolved_drugs"] = resolved
    return state


async def classify_query(state: AgentState) -> AgentState:
    """Classify the user's query into one of four categories via LLM.

    Args:
        state: Current agent state with original_query and resolved_drugs.

    Returns:
        Updated state with query_type set.
    """
    query = state["original_query"]
    drug_names = [d["generic_name"] for d in state["resolved_drugs"]]

    try:
        prompt = (
            "Classify the following drug-related question into exactly one category.\n\n"
            "Categories:\n"
            "- interaction_check: asking about drug-drug interactions\n"
            "- side_effect: asking about side effects or adverse reactions\n"
            "- contraindication: asking about when a drug should NOT be used\n"
            "- general_info: general drug information, indications, or usage\n\n"
            f"Drugs mentioned: {', '.join(drug_names)}\n"
            f"Question: {query}\n\n"
            "Respond with ONLY the category name, nothing else."
        )
        response = await _llm.ainvoke(prompt)
        category = response.content.strip().lower()

        valid = {"interaction_check", "side_effect", "contraindication", "general_info"}
        state["query_type"] = category if category in valid else "general_info"
    except Exception as exc:
        logger.error("Query classification failed: %s", exc)
        state["query_type"] = "general_info"

    return state


_QUERY_TYPE_CONTEXT = {
    "interaction_check": "drug interactions between",
    "side_effect": "side effects and adverse reactions of",
    "contraindication": "contraindications for",
    "general_info": "information about",
}


async def retrieve_from_index(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from the FAISS+BM25 index.

    Args:
        state: Current agent state with resolved_drugs and query_type.

    Returns:
        Updated state with retrieved_chunks populated.
    """
    drug_names = [d["generic_name"] for d in state["resolved_drugs"]]
    context_phrase = _QUERY_TYPE_CONTEXT.get(state["query_type"], "information about")
    query = f"{context_phrase} {' '.join(drug_names)} {state['original_query']}"

    try:
        chunks = await retriever.retrieve(query, drug_names, top_k=5)
        state["retrieved_chunks"] = chunks
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        state["retrieved_chunks"] = []

    return state


async def generate_response(state: AgentState) -> AgentState:
    """Generate the final answer using retrieved chunks as context.

    Args:
        state: Current agent state with retrieved_chunks and query metadata.

    Returns:
        Updated state with answer and citations populated.
    """
    chunks = state["retrieved_chunks"]

    if not chunks:
        state["answer"] = (
            "I could not find sufficient information in the FDA database to "
            "answer your question. Please try rephrasing or consult your "
            "healthcare provider." + _DISCLAIMER
        )
        state["citations"] = []
        return state

    # Build context block with source numbers
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        header = (
            f"[Source {i}] Drug: {meta.get('drug_generic_name', 'unknown')} | "
            f"Section: {meta.get('section_type', 'unknown')} | "
            f"Route: {meta.get('route', 'UNKNOWN')}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_parts)
    drug_names = [d["generic_name"] for d in state["resolved_drugs"]]

    prompt = (
        "You are a pharmaceutical information assistant. Answer the user's question "
        "using ONLY the provided FDA label excerpts.\n\n"
        "STRICT RULES — follow every one:\n"
        "1. Every factual sentence MUST end with a [Source N] citation. "
        "Never make a factual statement without a citation.\n"
        "2. Do NOT paraphrase — stay close to the source text wording.\n"
        "3. If you cannot support a claim from the provided sources, do not include it.\n"
        "4. Do NOT include any information not found in the provided sources.\n"
        "5. If the sources do not contain enough information, say so explicitly.\n"
        "6. NEVER recommend specific dosages.\n"
        "7. End your response with the disclaimer: "
        '"⚠️ This information is for educational purposes only. '
        'Always consult your healthcare provider or pharmacist."\n\n'
        "EXAMPLE of correct citation format:\n"
        '"Ibuprofen may increase the risk of bleeding when taken with warfarin [Source 1]. '
        "NSAIDs can interfere with the anticoagulant effect of warfarin [Source 2]. "
        'Concurrent use should be approached with caution [Source 1]."\n\n'
        "Notice: every factual sentence ends with [Source N]. Follow this pattern exactly.\n\n"
        "IMPORTANT — when no drug interaction is found between the queried drugs, you must "
        "STILL cite every factual statement about each drug. For example:\n"
        '"Acetaminophen is indicated for the temporary relief of minor aches and pains [Source 1]. '
        "Loratadine is used for the relief of symptoms associated with allergic rhinitis [Source 3]. "
        "No specific interaction between acetaminophen and loratadine was found in the available "
        'FDA labeling [Source 1][Source 3]."\n'
        "Do NOT describe any drug's properties, uses, or warnings without a [Source N] citation, "
        "even if the main finding is that no interaction exists.\n\n"
        f"Drugs involved: {', '.join(drug_names)}\n"
        f"Question: {state['original_query']}\n\n"
        f"Sources:\n{context_block}"
    )

    try:
        response = await _llm.ainvoke(prompt)
        answer = response.content.strip()
        state["answer"] = answer

        # Extract citations from the answer
        citations: list[dict] = []
        source_refs = set(re.findall(r"\[Source (\d+)\]", answer))
        for ref_num in sorted(source_refs, key=int):
            idx = int(ref_num) - 1
            if 0 <= idx < len(chunks):
                meta = chunks[idx]["metadata"]
                citations.append({
                    "claim": f"Source {ref_num}",
                    "source_type": "openFDA",
                    "drug_name": meta.get("drug_generic_name", "unknown"),
                    "section_type": meta.get("section_type", "unknown"),
                })
        state["citations"] = citations
    except Exception as exc:
        logger.error("Response generation failed: %s", exc)
        state["answer"] = (
            "An error occurred while generating the response. "
            "Please try again." + _DISCLAIMER
        )
        state["citations"] = []

    return state


async def output_safety_check(state: AgentState) -> AgentState:
    """Post-process the answer: ensure disclaimer, verify citations, strip dosage.

    Args:
        state: Current agent state with answer and citations.

    Returns:
        Updated state with sanitized answer.
    """
    answer = state["answer"]

    # Strip dosage recommendations
    answer = _DOSAGE_PATTERN.sub("[specific dosage removed — consult your provider]", answer)

    # Verify citations exist — if answer references sources but citations list is empty, note it
    source_refs = re.findall(r"\[Source \d+\]", answer)
    if source_refs and not state["citations"]:
        logger.error("Answer contains source references but no citations were extracted.")

    # Append disclaimer if not already present
    if "\u26a0\ufe0f" not in answer:
        answer += _DISCLAIMER

    state["answer"] = answer

    # Build formatted citation list from retrieved chunks
    chunks = state["retrieved_chunks"]
    if chunks:
        lines = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            drug = meta.get("drug_generic_name", "unknown")
            section = meta.get("section_type", "unknown")
            lines.append(f"[Source {i}] {drug} | {section} (openFDA)")
        state["sources_text"] = "\n".join(lines)
    else:
        state["sources_text"] = ""

    return state
