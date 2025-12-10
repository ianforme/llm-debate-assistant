# -*- coding: utf-8 -*-
"""
Opening statement workflow graph.

Deterministic workflow that leverages topic_research output:
1. load_research - Load topic_research results from filesystem
2. select_strategy - Select key terms + 3 arguments strategically
3. deep_evidence - Deep search (8-12 sources per argument)
4. draft - Create opening statement (4-min, max 1200 chars)
5. evaluate - Evaluate quality (pass/fail + feedback)
6. improve - Improve based on feedback (if needed, loop back to evaluate)
"""

from typing import Literal

from langgraph.graph import StateGraph, END

from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.schema import (
    OpeningState,
)
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.operations import (
    load_research_node,
    select_strategy_node,
    deep_evidence_node,
    draft_statement_node,
    evaluate_statement_node,
    improve_statement_node,
)


def route_evaluation(state: OpeningState) -> Literal["improve", "end"]:
    """Route after evaluation: improve if failed, end if passed or max iterations.
    Args:
        state (OpeningState): Current state of the opening statement workflow

    Returns:
        Literal["improve", "end"]: Next state to transition to
    """
    evaluation = state.get("evaluation")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    # Pass → done
    if evaluation and evaluation.get("result") == "pass":
        return "end"

    # Fail but reached max iterations → done (accept current draft)
    if iteration_count >= max_iterations:
        return "end"

    # Fail and can retry → improve
    return "improve"


def create_opening_graph() -> StateGraph:
    """Create opening statement workflow graph.

    Returns:
        StateGraph: Compiled StateGraph for opening statement generation
    """
    workflow = StateGraph(OpeningState)

    # Add all nodes
    workflow.add_node("load_research", load_research_node)
    workflow.add_node("select_strategy", select_strategy_node)
    workflow.add_node("deep_evidence", deep_evidence_node)
    workflow.add_node("draft", draft_statement_node)
    workflow.add_node("evaluate", evaluate_statement_node)
    workflow.add_node("improve", improve_statement_node)

    # Build predetermined flow
    workflow.set_entry_point("load_research")
    workflow.add_edge("load_research", "select_strategy")
    workflow.add_edge("select_strategy", "deep_evidence")
    workflow.add_edge("deep_evidence", "draft")
    workflow.add_edge("draft", "evaluate")

    # Evaluation loop: improve if failed, end if passed
    workflow.add_conditional_edges(
        "evaluate",
        route_evaluation,
        {
            "improve": "improve",
            "end": END,
        },
    )

    # After improvement, evaluate again
    workflow.add_edge("improve", "evaluate")

    return workflow
