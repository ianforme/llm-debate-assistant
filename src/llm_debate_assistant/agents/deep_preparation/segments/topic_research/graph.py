# -*- coding: utf-8 -*-
"""
LangGraph workflow for topic research .

This module provides a graph-based implementation of topic research that
executes automatically without human intervention.

Remark: Consider implementing HITL for verification of key steps in future iterations.

Usage:
    >>> # Create and run workflow
    >>> workflow = create_topic_research_graph()
    >>> app = workflow.compile()
    >>>
    >>> # Run to completion
    >>> config = {"configurable": {"thread_id": "research-123"}}
    >>> result = await app.ainvoke(initial_state, config)
"""

import asyncio
from typing import Any, Dict, Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    KeyTerm,
    PerspectiveResearch,
    ComparativeAnalysis,
    TopicResearchResult,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations.define_terms import (
    define_key_terms,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations.research_side import (
    research_perspective,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations.compare import (
    comparative_analysis,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.storage import (
    save_research_to_filesystem,
    load_research_from_filesystem,
    save_key_terms_progress,
    save_research_progress,
    save_analysis_progress,
    load_research_progress,
)
from llm_debate_assistant.agents.deep_preparation.schema import Filesystem


# ============================================================================
# State Definition
# ============================================================================


class TopicResearchState(TypedDict, total=False):
    """State for topic research graph."""

    # Context inputs
    topic: str
    side: Literal["正方", "反方"]
    filesystem: Optional[Filesystem]
    use_cache: bool

    # Research outputs
    key_terms: Optional[list[KeyTerm]]
    our_research: Optional[PerspectiveResearch]
    opponent_research: Optional[PerspectiveResearch]
    analysis: Optional[ComparativeAnalysis]
    research_result: Optional[TopicResearchResult]


# ============================================================================
# Graph Nodes
# ============================================================================


async def check_cache_node(state: TopicResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Check if cached research exists and load it (full or partial).

    Args:
        state (TopicResearchState): Current state of the research
        config (RunnableConfig): Runnable configuration for LLM calls

    Returns:
        Dict[str, Any]: returns any loaded cached data or empty dict if none found
    """
    if not state.get("use_cache", True) or not state.get("filesystem"):
        return {}

    filesystem = state["filesystem"]
    assert filesystem is not None  # Type narrowing: already checked above

    # First check for complete research
    cached = load_research_from_filesystem(filesystem)
    if cached:
        return {
            "research_result": cached,
            "key_terms": cached.key_terms,
            "our_research": cached.our_research,
            "opponent_research": cached.opponent_research,
            "analysis": cached.analysis,
        }

    # If no complete research, check for partial progress
    progress = load_research_progress(filesystem)
    if progress:
        from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
            KeyTerm,
            PerspectiveResearch,
            ComparativeAnalysis,
        )

        stage = progress.get("stage")
        result: Dict[str, Any] = {}

        # Load what we have based on stage
        if stage in ["terms_complete", "research_complete", "analysis_complete"]:
            result["key_terms"] = [KeyTerm(**t) for t in progress.get("key_terms", [])]

        if stage in ["research_complete", "analysis_complete"]:
            result["our_research"] = PerspectiveResearch(**progress["our_research"])
            result["opponent_research"] = PerspectiveResearch(**progress["opponent_research"])

        if stage == "analysis_complete":
            result["analysis"] = ComparativeAnalysis(**progress["analysis"])

        return result

    return {}


async def terms_definition_node(
    state: TopicResearchState, config: RunnableConfig
) -> Dict[str, Any]:
    """Generate key terms.

    Args:
        state (TopicResearchState): Current state of the research
        config (RunnableConfig): Runnable configuration for LLM calls

    Returns:
        Dict[str, Any]: returns generated key terms or empty dict if already present
    """
    # Skip if already loaded from cache
    if state.get("key_terms"):
        return {}

    topic = state["topic"]
    our_side = state["side"]

    # Generate terms
    terms = await define_key_terms(topic, our_side, config)

    # Save progress for resumability
    filesystem = state.get("filesystem")
    if filesystem is not None:
        save_key_terms_progress(topic, our_side, terms, filesystem)

    return {"key_terms": terms}


async def research_node(state: TopicResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Research both sides with full arguments and evidence.

    Args:
        state (TopicResearchState): Current state of the research
        config (RunnableConfig): Runnable configuration for LLM calls

    Returns:
        Dict[str, Any]: returns research results for both sides or empty dict if already present
    """
    # Skip if already loaded from cache
    if state.get("our_research") and state.get("opponent_research"):
        return {}

    topic = state["topic"]
    our_side = state["side"]
    opponent_side: Literal["正方", "反方"] = "反方" if our_side == "正方" else "正方"

    # Research both sides in parallel
    our_research, opponent_research = await asyncio.gather(
        research_perspective(topic, our_side, config),
        research_perspective(topic, opponent_side, config),
    )

    # Save progress for resumability
    filesystem = state.get("filesystem")
    key_terms = state.get("key_terms")
    if filesystem is not None and key_terms is not None:
        save_research_progress(
            topic,
            our_side,
            key_terms,
            our_research,
            opponent_research,
            filesystem,
        )

    return {
        "our_research": our_research,
        "opponent_research": opponent_research,
    }


async def analysis_node(state: TopicResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Perform comparative analysis.

    Args:
        state (TopicResearchState): Current state of the research
        config (RunnableConfig): Runnable configuration for LLM calls
    Returns:
        Dict[str, Any]: returns comparative analysis result or empty dict if already present
    """
    # Skip if already loaded from cache
    if state.get("analysis"):
        return {}

    our_research = state["our_research"]
    opponent_research = state["opponent_research"]
    assert (
        our_research is not None and opponent_research is not None
    )  # Should be set by research_node

    # Perform comparative analysis
    analysis_result = await comparative_analysis(our_research, opponent_research, config)

    # Save progress for resumability
    filesystem = state.get("filesystem")
    key_terms = state.get("key_terms")
    if filesystem is not None and key_terms is not None:
        save_analysis_progress(
            state["topic"],
            state["side"],
            key_terms,
            our_research,
            opponent_research,
            analysis_result,
            filesystem,
        )

    return {"analysis": analysis_result}


async def finalize_node(state: TopicResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Assemble final result and save to filesystem.

    Args:
        state (TopicResearchState): Current state of the research
        config (RunnableConfig): Runnable configuration for LLM calls

    Returns:
        Dict[str, Any]: returns final research result or empty dict if already present
    """
    # Check if we loaded from cache
    if state.get("research_result"):
        return {}  # Already have final result

    # These should all be populated by previous nodes
    key_terms = state["key_terms"]
    our_research = state["our_research"]
    opponent_research = state["opponent_research"]
    analysis = state["analysis"]
    assert (
        key_terms is not None
        and our_research is not None
        and opponent_research is not None
        and analysis is not None
    ), "All research components must be completed before finalize"

    # Assemble complete result
    result = TopicResearchResult(
        topic=state["topic"],
        our_side=state["side"],
        key_terms=key_terms,
        our_research=our_research,
        opponent_research=opponent_research,
        analysis=analysis,
    )

    # Save to filesystem if provided
    filesystem = state.get("filesystem")
    if filesystem is not None:
        save_research_to_filesystem(result, filesystem)

    return {"research_result": result}


# ============================================================================
# Routing Functions
# ============================================================================


def route_after_cache(state: TopicResearchState) -> str:
    """Determine the next step after checking cache.

    Intelligently routes based on what's cached:
    - Complete result → finalize
    - Has analysis → finalize (just needs packaging)
    - Has both research sides → analysis
    - Has key terms → research
    - Has nothing → terms_definition

    Args:
        state (TopicResearchState): Current state of the research

    Returns:
        str: Next node to transition to
    """
    # Complete result cached
    if state.get("research_result"):
        return "finalize"

    # Analysis complete but not packaged yet
    if state.get("analysis"):
        return "finalize"

    # Research complete, need analysis
    if state.get("our_research") and state.get("opponent_research"):
        return "analysis"

    # Terms defined, need research
    if state.get("key_terms"):
        return "research"

    # Nothing cached, start from beginning
    return "terms_definition"


# ============================================================================
# Graph Construction
# ============================================================================


def create_topic_research_graph() -> StateGraph:
    """Create the topic research workflow graph.

    Graph structure:
        START → check_cache → [route based on cached state]
            ├─ Complete result → finalize
            ├─ Has analysis → finalize
            ├─ Has research → analysis → finalize
            ├─ Has terms → research → analysis → finalize
            └─ Nothing → terms_definition → research → analysis → finalize
        All paths → END

    The graph supports intelligent resumption from partial cache states,
    skipping already-completed stages for efficiency.

    Returns:
        StateGraph ready for compilation

    Example:
        >>> from llm_debate_assistant.agents.deep_preparation.storage import DiskFilesystem
        >>>
        >>> graph = create_topic_research_graph()
        >>> app = graph.compile()
        >>>
        >>> filesystem = DiskFilesystem("./output")
        >>> state = create_initial_state(
        ...     topic="人工智能的发展利大于弊",
        ...     side="正方",
        ...     filesystem=filesystem,
        ...     use_cache=True,  # Enable smart resumption
        ... )
        >>> result = await app.ainvoke(state, {})
    """
    workflow = StateGraph(TopicResearchState)

    # Add nodes
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("terms_definition", terms_definition_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("check_cache")

    # Build edges - cache can route to any stage
    workflow.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {
            "finalize": "finalize",
            "analysis": "analysis",
            "research": "research",
            "terms_definition": "terms_definition",
        },
    )

    # Linear flow for uncached execution
    workflow.add_edge("terms_definition", "research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "finalize")
    workflow.add_edge("finalize", END)

    return workflow


# ============================================================================
# Convenience Functions
# ============================================================================


def create_initial_state(
    topic: str,
    side: Literal["正方", "反方"],
    filesystem: Optional[Filesystem] = None,
    use_cache: bool = True,
) -> TopicResearchState:
    """Create initial state for topic research graph.

    Args:
        topic: Debate topic
        side: Our side (正方 or 反方)
        filesystem: Optional filesystem for storing results
        use_cache: Whether to check for cached results

    Returns:
        Initial state dictionary
    """
    return {
        "topic": topic,
        "side": side,
        "filesystem": filesystem,
        "use_cache": use_cache,
        # Research outputs (will be populated)
        "key_terms": None,
        "our_research": None,
        "opponent_research": None,
        "analysis": None,
        "research_result": None,
    }
