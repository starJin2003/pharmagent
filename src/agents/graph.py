"""LangGraph workflow definition for the PharmAgent pipeline."""

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    classify_query,
    generate_response,
    input_safety_check,
    output_safety_check,
    resolve_drugs,
    retrieve_from_index,
)
from src.agents.state import AgentState


def _route_after_safety(state: AgentState) -> str:
    """Route based on safety check result.

    Args:
        state: Current agent state.

    Returns:
        Next node name or END.
    """
    if state["safety_flag"] != "ok":
        return END
    return "resolve_drugs"


def _route_after_resolve(state: AgentState) -> str:
    """Route based on drug resolution result.

    Args:
        state: Current agent state.

    Returns:
        Next node name or END.
    """
    if not state["resolved_drugs"]:
        return END
    return "classify_query"


def _build_graph() -> StateGraph:
    """Build and compile the PharmAgent LangGraph workflow.

    Returns:
        Compiled graph ready for ainvoke().
    """
    builder = StateGraph(AgentState)

    # Add all nodes
    builder.add_node("input_safety_check", input_safety_check)
    builder.add_node("resolve_drugs", resolve_drugs)
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_from_index", retrieve_from_index)
    builder.add_node("generate_response", generate_response)
    builder.add_node("output_safety_check", output_safety_check)

    # Wire edges
    builder.add_edge(START, "input_safety_check")
    builder.add_conditional_edges("input_safety_check", _route_after_safety)
    builder.add_conditional_edges("resolve_drugs", _route_after_resolve)
    builder.add_edge("classify_query", "retrieve_from_index")
    builder.add_edge("retrieve_from_index", "generate_response")
    builder.add_edge("generate_response", "output_safety_check")
    builder.add_edge("output_safety_check", END)

    return builder.compile()


graph = _build_graph()
