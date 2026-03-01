"""Gradio app for PharmAgent drug interaction queries."""

import gradio as gr

from src.agents.graph import graph


async def _handle_query(query: str) -> str:
    """Process a user query through the PharmAgent pipeline.

    Args:
        query: The user's drug-related question.

    Returns:
        The agent's answer with citations and disclaimer.
    """
    if not query or not query.strip():
        return "Please enter a drug-related question."

    initial_state = {
        "original_query": query.strip(),
        "safety_flag": "",
        "safety_message": None,
        "resolved_drugs": [],
        "query_type": "",
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
    }

    result = await graph.ainvoke(initial_state)

    if result["safety_flag"] != "ok":
        return result["safety_message"] or "Unable to process this query."

    if not result["resolved_drugs"]:
        return (
            "I couldn't identify any drug names in your question. "
            "Please try rephrasing with specific drug names (brand or generic)."
        )

    return result["answer"]


_EXAMPLES = [
    ["Can I take ibuprofen and warfarin together?"],
    ["What are the side effects of metformin?"],
    ["Is aspirin safe with lisinopril?"],
]

with gr.Blocks(title="PharmAgent — FDA Drug Interaction Assistant") as demo:
    gr.Markdown(
        "# PharmAgent\n"
        "Ask questions about drug interactions, side effects, and contraindications. "
        "Answers are sourced from FDA-approved drug labels with citations."
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g. Can I take ibuprofen and warfarin together?",
            lines=2,
        )

    submit_btn = gr.Button("Ask PharmAgent", variant="primary")

    answer_output = gr.Textbox(
        label="Answer",
        lines=12,
        interactive=False,
    )

    gr.Examples(
        examples=_EXAMPLES,
        inputs=query_input,
    )

    submit_btn.click(fn=_handle_query, inputs=query_input, outputs=answer_output)
    query_input.submit(fn=_handle_query, inputs=query_input, outputs=answer_output)
