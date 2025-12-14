# -*- coding: utf-8 -*-
"""
Constructive speech workflow graph.
"""

import json
import logging
from typing import Any, Dict, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.schema import (
    ConstructiveState,
)
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.operations import (
    load_research,
    generate_constructive_strategy,
    deep_evidence_search,
    draft_constructive_speech,
    critique_constructive_speech,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Node Wrappers
# ============================================================================


async def load_research_node(state: ConstructiveState, config: RunnableConfig) -> Dict[str, Any]:
    """Load topic_research results (node wrapper).

    Args:
        state (ConstructiveState): Current state containing topic and side.
        config (RunnableConfig): Configuration.

    Returns:
        Dict[str, Any]: Update to state with 'research_context'.
    """
    research_result = load_research(
        topic=state["topic"],
        side=state["side"],
        research_context=state.get("research_context"),
    )

    # Only return update if we loaded new research
    if state.get("research_context") is None:
        return {"research_context": research_result}
    return {}


async def generate_strategy_node(
    state: ConstructiveState, config: RunnableConfig
) -> Dict[str, Any]:
    """Generate constructive strategy (node wrapper).

    Args:
        state (ConstructiveState): Current state.
        config (RunnableConfig): Configuration containing filesystem.

    Returns:
        Dict[str, Any]: Update to state with 'constructive_strategy'.
    """
    filesystem = config.get("configurable", {}).get("filesystem")
    assert filesystem is not None, "Filesystem is required"

    research_context = state["research_context"]
    assert research_context is not None, "Research context is required"

    strategy = await generate_constructive_strategy(
        topic=state["topic"],
        side=state["side"],
        research=research_context,
    )

    # Save strategy to filesystem
    filesystem.write(
        "/constructive_speech/strategy.json",
        strategy.model_dump_json(indent=2, ensure_ascii=False),
    )

    return {"constructive_strategy": strategy}


async def deep_evidence_node(state: ConstructiveState, config: RunnableConfig) -> Dict[str, Any]:
    """Perform deep evidence search (node wrapper).

    Args:
        state (ConstructiveState): Current state.
        config (RunnableConfig): Configuration containing filesystem.

    Returns:
        Dict[str, Any]: Update to state with 'deep_evidence'.
    """
    filesystem = config.get("configurable", {}).get("filesystem")
    assert filesystem is not None, "Filesystem is required"

    strategy = state.get("constructive_strategy")
    assert strategy is not None, "Strategy is required"

    evidence_list = await deep_evidence_search(
        topic=state["topic"],
        side=state["side"],
        selected_arguments=strategy.selected_arguments,
        config=config,
    )

    # Save evidence to filesystem
    evidence_dump = [ev.model_dump() for ev in evidence_list]
    filesystem.write(
        "/constructive_speech/deep_evidence.json",
        json.dumps(evidence_dump, indent=2, ensure_ascii=False),
    )

    return {"deep_evidence": evidence_list}


async def draft_node(state: ConstructiveState, config: RunnableConfig) -> Dict[str, Any]:
    """Draft the constructive speech (node wrapper).

    Args:
        state (ConstructiveState): Current state.
        config (RunnableConfig): Configuration containing filesystem.

    Returns:
        Dict[str, Any]: Update to state with 'draft_content' and 'iteration_count'.
    """
    filesystem = config.get("configurable", {}).get("filesystem")
    assert filesystem is not None, "Filesystem is required"

    strategy = state.get("constructive_strategy")
    assert strategy is not None, "Strategy is required"

    iteration = state.get("iteration_count", 0)
    deep_evidence = state.get("deep_evidence") or []

    draft_text = await draft_constructive_speech(
        topic=state["topic"],
        side=state["side"],
        strategy=strategy,
        deep_evidence=deep_evidence,
        iteration=iteration,
        critique=state.get("critique"),
    )

    # Save draft to filesystem
    filename = f"draft_v{iteration + 1}.md"
    filesystem.write(f"/constructive_speech/{filename}", draft_text)

    return {"draft_content": draft_text, "iteration_count": iteration + 1}


async def critique_node(state: ConstructiveState, config: RunnableConfig) -> Dict[str, Any]:
    """Critique the constructive speech (node wrapper).

    Args:
        state (ConstructiveState): Current state.
        config (RunnableConfig): Configuration.

    Returns:
        Dict[str, Any]: Update to state with 'critique'.
    """
    draft = state.get("draft_content")
    strategy = state.get("constructive_strategy")

    assert draft is not None, "Draft is required"
    assert strategy is not None, "Strategy is required"

    deep_evidence = state.get("deep_evidence") or []

    critique_result = await critique_constructive_speech(
        topic=state["topic"],
        side=state["side"],
        draft=draft,
        strategy=strategy,
        deep_evidence=deep_evidence,
    )

    return {"critique": critique_result}


def route_critique(
    state: ConstructiveState,
) -> Literal["draft_constructive_speech", "finalize_constructive_speech"]:
    """Decide the next step based on critique result.

    Routes to:
    - Pass -> Finalize
    - Fail + Max Retries -> Finalize (Cut losses)
    - Fail + Can Retry -> Draft (Loop back with feedback)

    Args:
        state (ConstructiveState): Current state.

    Returns:
        Literal["draft_constructive_speech", "finalize_constructive_speech"]: Next node name.
    """
    critique = state.get("critique")
    iteration = state.get("iteration_count", 0)

    max_retries = 3

    if not critique:
        return "finalize_constructive_speech"

    if critique.decision == "pass":
        return "finalize_constructive_speech"

    if iteration >= max_retries:
        return "finalize_constructive_speech"

    return "draft_constructive_speech"


async def finalize_constructive_speech_node(
    state: ConstructiveState, config: RunnableConfig
) -> Dict[str, Any]:
    """Finalize the constructive speech generation process.

    Tasks:
    1. Save the current draft as the official 'final_speech.md'.
    2. Save execution metadata (score, iterations, strategy summary).
    3. Prepare the final output payload.

    Args:
        state (ConstructiveState): The final state after the loop ends.
        config (RunnableConfig): Configuration containing filesystem.

    Returns:
        Dict[str, Any]: The final output of this subgraph.
    """
    draft = state.get("draft_content")
    strategy = state.get("constructive_strategy")
    critique = state.get("critique")
    iteration = state.get("iteration_count", 0)
    filesystem = config.get("configurable", {}).get("filesystem")

    assert draft is not None, "No draft content found at finalization"
    assert filesystem is not None

    logger.info(f"Finalizing Constructive Speech (Iterations: {iteration})")

    filesystem.write("/constructive_speech/final_speech.md", draft)

    final_status = "approved" if critique and critique.decision == "pass" else "max_retries_reached"
    final_score = critique.score if critique else 0

    metadata = {
        "status": final_status,
        "final_score": final_score,
        "iterations": iteration,
        "strategy_summary": {
            "tone": strategy.speech_tone if strategy else "N/A",
            "value_premise": strategy.value_premise if strategy else "N/A",
            "arguments": ([arg.claim for arg in strategy.selected_arguments] if strategy else []),
        },
        "outstanding_issues": (
            critique.critical_issues if critique and final_status != "approved" else []
        ),
    }

    filesystem.write(
        "/constructive_speech/meta.json",
        json.dumps(metadata, indent=2, ensure_ascii=False),
    )

    logger.info(f"Artifacts saved. Final Score: {final_score}/10. Status: {final_status}")

    return {
        "final_speech_content": draft,
        "final_metadata": metadata,
        "is_complete": True,
    }


def create_constructive_graph() -> StateGraph:
    """Create the constructive speech generation workflow.

    Workflow:
    1. Load Context (Research)
    2. Generate Strategy (Hook/Pivot/Anchor)
    3. Deep Search (Evidence)
    4. Write Draft <---|
    5. Critique -------| (Loop if needs revision)
    6. Finalize

    Returns:
        StateGraph: Compiled graph.
    """
    workflow = StateGraph(ConstructiveState)

    workflow.add_node("load_research", load_research_node)
    workflow.add_node("generate_strategy", generate_strategy_node)
    workflow.add_node("deep_evidence", deep_evidence_node)

    workflow.add_node("draft_constructive_speech", draft_node)
    workflow.add_node("critique_constructive_speech", critique_node)

    workflow.add_node("finalize_constructive_speech", finalize_constructive_speech_node)

    workflow.set_entry_point("load_research")
    workflow.add_edge("load_research", "generate_strategy")
    workflow.add_edge("generate_strategy", "deep_evidence")
    workflow.add_edge("deep_evidence", "draft_constructive_speech")

    workflow.add_edge("draft_constructive_speech", "critique_constructive_speech")

    workflow.add_conditional_edges(
        "critique_constructive_speech",
        route_critique,
        {
            "draft_constructive_speech": "draft_constructive_speech",
            "finalize_constructive_speech": "finalize_constructive_speech",
        },
    )

    workflow.add_edge("finalize_constructive_speech", END)

    return workflow


if __name__ == "__main__":
    """Generate and save graph visualization."""
    import os

    # Create and compile workflow
    workflow = create_constructive_graph()
    app = workflow.compile()

    # Generate PNG visualization
    output_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../../asset/constructive_speech_graph.png",
    )

    # Ensure asset directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Draw and save
    png_data = app.get_graph(xray=True).draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_data)

    print(f"Graph visualization saved to: {output_path}")
