"""
Agent-driven deep preparation workflow graph.

This module creates a LangGraph workflow where an agent autonomously decides
which tools to call and when, rather than following a predetermined sequence.

Workflow:
    START → init_filesystem → agent ⇄ tools → END
                                ↑_____|

The agent loop continues until the agent decides it's done (no more tool calls).
"""

from functools import partial
from typing import Any, Literal, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from llm_debate_assistant.agents.deep_preparation.segments.opening.agent_nodes import (  # type: ignore[import-untyped]
    agent_node,
    tool_execution_node,
    should_continue,
)
from llm_debate_assistant.agents.deep_preparation.storage import (
    initialize_filesystem_node,
)
from llm_debate_assistant.agents.deep_preparation.schema import DeepPrepState


def create_deep_prep_workflow(
    filesystem_type: Literal["virtual", "disk"] = "disk",
    session_id: Optional[str] = None,
) -> CompiledStateGraph[Any, Any]:
    """Create the agent-driven deep preparation workflow.

    This workflow uses an agent that autonomously decides which tools to call
    and when to use them, rather than following a predetermined sequence.

    The agent has access to:
    - Core workflow tools: create_outline, search_evidence, draft_statement,
      evaluate_statement, improve_statement
    - Support tools: write_file, read_file, write_todos

    Args:
        filesystem_type (Literal["virtual", "disk"]): Type of filesystem to use:
            - "virtual": In-memory, transient storage (no disk I/O)
            - "disk": Persistent storage in
                .temp/debate_preparation_sessions/<session_id>/
        session_id (Optional[str]): Optional session ID (only used for disk filesystem)

    Returns:
        StateGraph: Compiled LangGraph workflow

    Example:
        >>> workflow = create_deep_prep_workflow(filesystem_type="disk")
        >>> initial_state = {
        ...     "topic": "人工智能的发展利大于弊",
        ...     "side": "正方",
        ...     "filesystem": None,
        ...     "filesystem_path": None,
        ...     "outline": None,
        ...     "evidence_data": None,
        ...     "draft": None,
        ...     "evaluation": None,
        ...     "todos": None,
        ...     "iteration_count": 1,
        ...     "max_iterations": 3,
        ...     "next_action": "auto",
        ...     "messages": [
        ...         HumanMessage(content="准备这个辩题的开篇立论")
        ...     ],
        ... }
        >>> result = await workflow.ainvoke(initial_state)
    """
    workflow = StateGraph(DeepPrepState)

    # Add initialization node for filesystem with bound parameters
    init_node = partial(
        initialize_filesystem_node,
        filesystem_type=filesystem_type,
        session_id=session_id,
    )
    workflow.add_node("init_filesystem", init_node)

    # Add agent loop nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)

    # Set entry point
    workflow.set_entry_point("init_filesystem")

    # Build the graph
    # init_filesystem → agent
    workflow.add_edge("init_filesystem", "agent")

    # agent → (conditional: continue to tools OR end)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",  # Agent has tool calls to execute
            "end": END,  # Agent is done
        },
    )

    # tools → agent (loop back for next decision)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============================================================================
# Convenience function for creating initial state
# ============================================================================


def create_initial_state(
    topic: str,
    side: str,
    user_message: str = "准备这个辩题的开篇立论",
    max_iterations: int = 3,
    iteration_count: int = 1,
    verbose: bool = False,
) -> DeepPrepState:
    """Create initial state for deep preparation workflow.

    Args:
        topic (str): Debate topic
        side (str): Debate side (正方 or 反方)
        user_message (str): Initial message to the agent
        max_iterations (int): Maximum number of improvement iterations
        iteration_count (int): Starting iteration count
        verbose (bool): Enable verbose logging (message stats, etc.)

    Returns:
        DeepPrepState: Initial state dictionary
    """
    return {
        "topic": topic,
        "side": side,
        "filesystem": None,
        "filesystem_path": None,
        "outline": None,
        "evidence_data": None,
        "draft": None,
        "evaluation": None,
        "todos": None,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "next_action": "auto",
        "verbose": verbose,
        "messages": [HumanMessage(content=user_message)],
    }
