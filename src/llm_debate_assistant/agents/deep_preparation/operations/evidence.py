"""
Evidence search operation.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.services.web_search import search_multiple_arguments
from ..schema import DeepPrepState
from .helpers import load_outline_from_fs


async def search_evidence_node_fs(
    state: DeepPrepState, config: RunnableConfig
) -> dict[str, Any]:
    """Search for evidence supporting debate arguments (filesystem-aware).

    Reads outline from filesystem to save tokens.

    Args:
        state (DeepPrepState): Deep preparation agent state
        config (RunnableConfig): Runnable configuration

    Returns:
        dict[str, Any]: Updated state with evidence data
    """
    # Load outline from filesystem
    outline = load_outline_from_fs(state)
    arguments = outline["arguments"]

    # Prepare search arguments
    search_args = [
        (arg["claim"], arg["warrant"], arg["evidence_needed"]) for arg in arguments
    ]

    # Check if there's feedback from evaluation (when redoing evidence)
    feedback = None
    evaluation = state.get("evaluation")
    if evaluation and evaluation.get("feedback"):
        feedback = evaluation["feedback"]

    # Use thread-based concurrent search
    evidence_results = await search_multiple_arguments(
        arguments=search_args,
        topic=state["topic"],
        side=state["side"],
        use_threaded=True,
        feedback=feedback,
    )

    # Convert to dict format and store directly in state
    evidence_dicts = [
        {
            "argument": res.argument,
            "warrant": res.warrant,
            "search_results": res.results.get("search_results", []),
            "analysis": res.results.get("text", ""),
            "error": res.error,
        }
        for res in evidence_results
    ]

    return {
        "evidence_data": evidence_dicts,
    }
