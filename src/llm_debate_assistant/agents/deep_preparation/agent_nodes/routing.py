"""
Routing logic for the deep preparation workflow.
"""

from typing import Literal

from langchain_core.messages import AIMessage

from ..schema import DeepPrepState


def should_continue(state: DeepPrepState) -> Literal["continue", "end"]:
    """Decide whether to continue agent loop or end.

    Checks both message state and workflow completion state to determine
    if the workflow should continue or end.

    Args:
        state (DeepPrepState): Current agent state

    Returns:
        Literal["continue", "end"]: "continue" if agent has more tool calls
            to execute, "end" if done
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Check if workflow is complete based on state
    evaluation = state.get("evaluation")
    if evaluation:
        evaluation_result = evaluation.get("evaluation_result")
        current_iteration = evaluation.get("iteration_number", 1)
        max_iterations = state.get("max_iterations", 3)

        # Check if evaluation step is complete
        is_evaluation_complete = (
            evaluation_result == "pass" or current_iteration >= max_iterations
        )

        # Check if all required artifacts exist
        has_all_artifacts = (
            bool(state.get("outline"))
            and bool(state.get("evidence_data"))
            and bool(state.get("draft"))
        )

        # If evaluation is complete and we have all artifacts, workflow is done
        if is_evaluation_complete and has_all_artifacts:
            return "end"

    # Check if agent has more tool calls to execute
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"

    # Check if workflow is incomplete
    has_outline = bool(state.get("outline"))
    has_evidence = bool(state.get("evidence_data"))
    has_draft = bool(state.get("draft"))
    has_evaluation = bool(state.get("evaluation"))

    workflow_incomplete = not (
        has_outline and has_evidence and has_draft and has_evaluation
    )

    if workflow_incomplete:
        # Check todos for pending work
        todos = state.get("todos", [])
        pending_todos = (
            [t for t in todos if t.get("status") in ("pending", "in_progress")]
            if todos
            else []
        )

        if pending_todos or not todos:
            # End to avoid infinite loop
            return "end"

    return "end"
