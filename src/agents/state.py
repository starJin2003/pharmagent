"""Agent state definition for the PharmAgent workflow."""

from typing import Optional, TypedDict


class AgentState(TypedDict):
    """State passed between all nodes in the LangGraph workflow.

    Attributes:
        original_query: The user's raw input question.
        safety_flag: One of "ok", "emergency", "out_of_scope", "needs_doctor".
        safety_message: Message to return early when safety_flag != "ok".
        resolved_drugs: List of dicts with keys {input_name, generic_name, brand_names, rxcui}.
        query_type: One of "interaction_check", "side_effect", "contraindication", "general_info".
        retrieved_chunks: List of chunk dicts with keys {text, metadata, score}.
        answer: The final generated answer text.
        citations: List of citation dicts with keys {claim, source_type, drug_name, section_type}.
    """

    original_query: str
    safety_flag: str
    safety_message: Optional[str]
    resolved_drugs: list[dict]
    query_type: str
    retrieved_chunks: list[dict]
    answer: str
    citations: list[dict]
