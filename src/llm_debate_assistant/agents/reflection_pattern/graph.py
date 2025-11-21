from typing import Literal

from langgraph.graph import END, StateGraph

from .nodes import (
    create_outline_node,
    draft_statement_node,
    evaluate_statement_node,
    improve_statement_node,
    search_evidence_node,
)
from .state import DebateState

# evaluate_statement
#         ↓
# should_continue_improving()
#         ↓
#    (returns string)
#         ↓
#     ┌───┴────┬──────────┬──────────┬─────────┐
#     │        │          │          │         │
# "redo_    "redo_    "redo_     "improve"  "end"
# outline"  evidence"  draft"
#     │        │          │          │         │
#     ↓        ↓          ↓          ↓         ↓
# create_  search_   draft_    improve_     END
# outline  evidence  statement  statement


def should_continue_improving(
    state: DebateState,
) -> Literal["redo_outline", "redo_evidence", "redo_draft", "improve", "end"]:
    """Decide whether to improve, redo a specific step, or end.

    This supports both automatic improvement and manual redo of any step.

    Args:
        state (DebateState): Current debate state

    Returns:
        Next node to execute
    """
    # Check for manual redo request first
    next_action = state.get("next_action", "auto")

    if next_action == "redo_outline":
        return "redo_outline"
    elif next_action == "redo_evidence":
        return "redo_evidence"
    elif next_action == "redo_draft":
        return "redo_draft"

    # Otherwise, automatic improvement logic
    evaluation = state.get("evaluation") or {}
    iteration_count = state.get("iteration_count") or 1
    max_iterations = state.get("max_iterations") or 3

    # End if passed or max iterations reached
    if evaluation.get("evaluation_result") == "pass":
        return "end"

    # iteration_count starts at 1 and increments after each improvement
    # Stop when we reach max_iterations
    # (e.g., max_iterations=3 stops at iteration_count=3)
    if iteration_count >= max_iterations:
        return "end"

    return "improve"


def create_debate_workflow() -> StateGraph:
    """Create the debate preparation workflow graph.

    This workflow includes:
        - Outline creation
        - Evidence search
        - Drafting statements
        - Evaluating statements
        - Improving statements

    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    workflow = StateGraph(DebateState)

    # Add nodes (no wrapper needed - using RunnableConfig directly)
    workflow.add_node("create_outline", create_outline_node)
    workflow.add_node("search_evidence", search_evidence_node)
    workflow.add_node("draft_statement", draft_statement_node)
    workflow.add_node("evaluate_statement", evaluate_statement_node)
    workflow.add_node("improve_statement", improve_statement_node)

    # Set entry point
    workflow.set_entry_point("create_outline")

    # Add edges (linear flow until evaluation)
    workflow.add_edge("create_outline", "search_evidence")
    workflow.add_edge("search_evidence", "draft_statement")
    workflow.add_edge("draft_statement", "evaluate_statement")

    # Conditional edge: evaluate -> improve/redo/end
    workflow.add_conditional_edges(
        "evaluate_statement",
        should_continue_improving,
        {
            "redo_outline": "create_outline",  # Jump back to outline
            "redo_evidence": "search_evidence",  # Jump back to evidence
            "redo_draft": "draft_statement",  # Jump back to draft
            "improve": "improve_statement",  # Normal improvement
            "end": END,
        },
    )

    # After improvement, go back to evaluate
    workflow.add_edge("improve_statement", "evaluate_statement")

    return workflow.compile()
