# -*- coding: utf-8 -*-
"""
Tool wrappers for the deep preparation orchestrator.

This module wraps subgraphs (topic_research, constructive_speech) and utilities
(todo list) as LangChain tools that can be called by the orchestrator agent.
"""

import logging
from typing import Literal, Optional

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.agents.deep_preparation.schema import Filesystem, PrepMode
from llm_debate_assistant.agents.deep_preparation.segments.topic_research import (
    create_topic_research_graph,
    create_initial_state as create_research_state,
)
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech import (
    create_constructive_graph,
)
from llm_debate_assistant.services.manage_todo_list import (
    manage_todo_list,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Factory Functions
# ============================================================================
# Tools need access to filesystem from config, so we create them dynamically


def create_topic_research_tool(filesystem: Filesystem):
    """Create topic research tool with filesystem bound."""

    @tool
    async def run_topic_research(
        topic: str,
        side: Literal["正方", "反方"],
    ) -> str:
        """
        Execute comprehensive topic research for the debate.

        This tool researches both sides of the debate, defines key terms strategically,
        and provides comparative analysis with strategic recommendations.

        IMPORTANT: This should be the FIRST tool called in any debate preparation.

        Args:
            topic (str): The debate topic (e.g., "应该强制要求大型科技公司开源其核心算法")
            side (Literal["正方", "反方"]): Our side in the debate ("正方" for proposition, "反方" for opposition)

        Returns:
            Summary of research completion status and artifact paths
        """
        logger.info(f"Starting topic research: {topic} ({side})")

        graph = create_topic_research_graph()
        app = graph.compile()

        initial_state = create_research_state(
            topic=topic,
            side=side,
            use_cache=True,
        )

        config: RunnableConfig = {"configurable": {"filesystem": filesystem}}

        try:
            final_state = await app.ainvoke(initial_state, config)
            result = final_state.get("research_result")

            if result:
                num_terms = len(result.key_terms)
                num_our_args = len(result.our_research.arguments)
                num_opp_args = len(result.opponent_research.arguments)
                num_clashes = len(result.analysis.key_clashes)

                return (
                    f"✅ Topic research completed successfully.\n\n"
                    f"Summary:\n"
                    f"- Key terms defined: {num_terms}\n"
                    f"- Our arguments: {num_our_args}\n"
                    f"- Opponent arguments: {num_opp_args}\n"
                    f"- Key clashes identified: {num_clashes}\n\n"
                    f"Artifacts saved to:\n"
                    f"- /research/research.json\n"
                    f"- /research/research_summary.md\n"
                    f"- /research/key_terms.md\n"
                    f"- /research/our_arguments.md\n"
                    f"- /research/opponent_arguments.md\n"
                    f"- /research/analysis.md"
                )
            else:
                return "⚠️ Topic research completed but no result returned."

        except Exception as e:
            logger.error(f"Topic research failed: {e}")
            return f"❌ Topic research failed: {str(e)}"

    return run_topic_research


def create_constructive_speech_tool(filesystem: Filesystem):
    """Create constructive speech tool with filesystem bound."""

    @tool
    async def run_constructive_speech(
        topic: str,
        side: Literal["正方", "反方"],
    ) -> str:
        """
        Generate a constructive speech (立论稿) based on completed research.

        This tool creates a complete opening statement including:
        - Strategic argument selection
        - Deep evidence search
        - Draft writing with critique loop
        - Final polished speech

        IMPORTANT: run_topic_research MUST be completed first before calling this tool.

        Args:
            topic (str): The debate topic
            side (Literal["正方", "反方"]): Our side in the debate ("正方" or "反方")

        Returns:
            str: Summary of speech generation status and artifact paths
        """
        logger.info(f"Starting constructive speech: {topic} ({side})")

        graph = create_constructive_graph()
        app = graph.compile()

        # Initial state for constructive speech
        initial_state = {
            "topic": topic,
            "side": side,
            "research_context": None,  # Will be loaded from filesystem
            "constructive_strategy": None,
            "deep_evidence": None,
            "draft_content": None,
            "critique": None,
            "iteration_count": 0,
            "final_speech_content": None,
            "final_metadata": None,
            "is_complete": False,
        }

        config: RunnableConfig = {"configurable": {"filesystem": filesystem}}

        try:
            final_state = await app.ainvoke(initial_state, config)

            if final_state.get("is_complete"):
                metadata = final_state.get("final_metadata", {})
                iterations = metadata.get("iterations", 0)
                score = metadata.get("final_score", 0)
                status = metadata.get("status", "unknown")

                return (
                    f"✅ Constructive speech completed successfully.\n\n"
                    f"Summary:\n"
                    f"- Status: {status}\n"
                    f"- Final score: {score}/10\n"
                    f"- Iterations: {iterations}\n\n"
                    f"Artifacts saved to:\n"
                    f"- /constructive_speech/final_speech.md\n"
                    f"- /constructive_speech/strategy.json\n"
                    f"- /constructive_speech/deep_evidence.json\n"
                    f"- /constructive_speech/meta.json"
                )
            else:
                return "⚠️ Constructive speech generation incomplete."

        except Exception as e:
            logger.error(f"Constructive speech failed: {e}")
            return f"❌ Constructive speech failed: {str(e)}"

    return run_constructive_speech


def create_todo_tool(state_ref: dict):
    """
    Create todo management tool with state reference.

    Args:
        state_ref: A mutable dict containing {"state": DeepPrepState} that will be
            updated as the agent runs. This allows the tool to access current state.
    """

    @tool
    def update_todo(
        action: Literal[
            "create",
            "add",
            "mark_complete",
            "mark_in_progress",
            "get_current",
            "summary",
        ],
        task_id: Optional[str] = None,
        description: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> str:
        """
        Manage the preparation todo list to track progress.

        Use this tool to:
        - Create a new todo list at the start of preparation
        - Add tasks for each preparation stage
        - Update task status as you progress
        - Check current task or get a summary

        Args:
            action (Literal[
                "create",
                "add",
                "mark_complete",
                "mark_in_progress",
                "get_current",
                "summary",
            ]): The action to perform:
                - "create": Initialize a new todo list (requires goal)
                - "add": Add a new task (requires task_id and description)
                - "mark_complete": Mark a task as completed (requires task_id)
                - "mark_in_progress": Mark a task as in progress (requires task_id)
                - "get_current": Get the current task being worked on
                - "summary": Get a formatted summary of all tasks
            task_id (Optional[str]): Task identifier (e.g., "1", "research", "speech")
            description (Optional[str]): Task description (for "add" action)
            goal (Optional[str]): Overall goal description (for "create" action)

        Returns:
            str: Status message or summary
        """
        state = state_ref.get("state", {})

        try:
            result = manage_todo_list(
                action=action,
                state=state,
                task_id=task_id,
                description=description,
                goal=goal,
            )

            # Update the todo_list in state reference
            if result.get("todo_list"):
                state["todo_list"] = result["todo_list"]
                state_ref["state"] = state

            # Return appropriate response based on action
            if action == "summary":
                return result.get("data", "No todo list found.")
            elif action == "get_current":
                current = result.get("data")
                if current:
                    return f"Current task: [{current.id}] {current.description}"
                else:
                    return "No task currently in progress."
            else:
                return result.get("message", "Action completed.")

        except Exception as e:
            return f"❌ Todo operation failed: {str(e)}"

    return update_todo


# ============================================================================
# Tool Sets by Mode
# ============================================================================


def get_tools_for_mode(
    mode: PrepMode,
    filesystem: Filesystem,
    state_ref: dict,
) -> list:
    """
    Get the appropriate tool set based on preparation mode.

    Args:
        mode (PrepMode): "lite" for topic_research only, "full" for all segments
        filesystem (Filesystem): Filesystem instance for saving artifacts
        state_ref (dict): Mutable dict containing current state

    Returns:
        list: List of tools available for the given mode
    """
    # Always available tools
    todo_tool = create_todo_tool(state_ref)
    research_tool = create_topic_research_tool(filesystem)

    if mode == "lite":
        return [todo_tool, research_tool]

    # Full mode includes all segment tools
    speech_tool = create_constructive_speech_tool(filesystem)

    return [
        todo_tool,
        research_tool,
        speech_tool,
        # Future tools:
        # create_rebuttal_tool(filesystem),
        # create_tactical_exchange_tool(filesystem),
        # create_closing_tool(filesystem),
    ]


# ============================================================================
# Tool Names (for reference)
# ============================================================================

TOOL_NAMES = {
    "todo": "update_todo",
    "research": "run_topic_research",
    "speech": "run_constructive_speech",
}
