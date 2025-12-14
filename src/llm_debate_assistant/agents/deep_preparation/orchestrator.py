# -*- coding: utf-8 -*-
"""
Deep Preparation Orchestrator.

This module implements a ReAct-style agent that orchestrates debate preparation
by invoking subgraph tools (topic_research, constructive_speech, etc.) based on
the preparation mode (lite/full).

The agent uses a todo list to track progress and makes autonomous decisions about
which preparation stage to execute next.
"""

import logging
from typing import Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from llm_debate_assistant.agents.deep_preparation.schema import (
    DeepPrepState,
    Filesystem,
    PrepMode,
)
from llm_debate_assistant.agents.deep_preparation.tools import get_tools_for_mode
from llm_debate_assistant.agents.deep_preparation.storage import (
    create_filesystem,
    save_session_metadata,
)
from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)


# ============================================================================
# System Prompts (Fallbacks)
# ============================================================================
# NOTE: These prompts should be uploaded to Opik with the following names:
#   - DEEP_PREP_ORCHESTRATOR_TODO_INSTRUCTIONS
#   - DEEP_PREP_ORCHESTRATOR_LITE
#   - DEEP_PREP_ORCHESTRATOR_FULL
#
# The get_system_prompt() function will load from Opik and only use these
# as fallbacks if the prompt manager fails.
# ============================================================================

# Base instructions for todo list management (inspired by deepagents patterns)
TODO_INSTRUCTIONS = """
## Task Planning and Progress Tracking

You MUST use the `update_todo` tool to manage your work. This is critical for:
- Breaking down complex tasks into discrete, trackable steps
- Providing visibility into your progress
- Maintaining focus on the current objective

### Todo List Workflow

1. **Initialize**: At the start, create a todo list with `action="create"` and set the goal
2. **Plan**: Add all tasks upfront with `action="add"` so the full scope is visible
3. **Execute**: For each task:
   - Mark it `in_progress` BEFORE starting work
   - Execute the relevant tool
   - Mark it `completed` IMMEDIATELY after success
4. **Adapt**: If new tasks emerge during execution, add them to the list
5. **Report**: Use `action="summary"` to show final progress

### Important Rules
- Only ONE task should be `in_progress` at a time
- Mark tasks complete AS SOON as they finish - do not batch completions
- If a task fails, keep it `in_progress` and report the error
- Always show the summary at the end

"""

LITE_SYSTEM_PROMPT = (
    """You are an expert debate preparation assistant operating in LITE mode.

Your mission is to conduct comprehensive topic research for the given debate topic.
This mode focuses solely on research - no speech drafting.

"""
    + TODO_INSTRUCTIONS
    + """

## Available Tools

### `update_todo`
Manage your task list. Actions: create, add, mark_in_progress, mark_complete, summary

### `run_topic_research`
Execute comprehensive topic research. This will:
- Define key terms strategically (with definitions that favor our position)
- Research arguments for both sides (我方 and 对方)
- Perform comparative analysis to identify key clashes
- Generate strategic recommendations

## Execution Plan

For a debate topic with our side being 正方 or 反方:

1. Create todo list: `update_todo(action="create", goal="Topic Research: [topic]")`
2. Add task: `update_todo(action="add", task_id="research", description="Conduct topic research")`
3. Start task: `update_todo(action="mark_in_progress", task_id="research")`
4. Execute: `run_topic_research(topic="...", side="正方/反方")`
5. Complete task: `update_todo(action="mark_complete", task_id="research")`
6. Show summary: `update_todo(action="summary")`
7. Report results to the user with key findings

## Output Format

After completing research, provide a brief summary including:
- Number of key terms defined
- Number of arguments found for each side
- Key strategic insights
- Location of saved artifacts

Begin preparation now.
"""
)

FULL_SYSTEM_PROMPT = (
    """You are an expert debate preparation assistant operating in FULL mode.

Your mission is to complete comprehensive debate preparation including:
1. Topic Research - understand both sides deeply
2. Constructive Speech - draft the opening argument (立论稿)

"""
    + TODO_INSTRUCTIONS
    + """

## Available Tools

### `update_todo`
Manage your task list. Actions: create, add, mark_in_progress, mark_complete, summary

### `run_topic_research`
Execute comprehensive topic research. This will:
- Define key terms strategically
- Research arguments for both sides
- Perform comparative analysis
- Generate strategic recommendations

**MUST be completed before constructive speech.**

### `run_constructive_speech`
Generate a complete opening statement (立论稿). This will:
- Select the strongest arguments from research
- Perform deep evidence search for each argument
- Draft the speech with critique loop for quality
- Produce a polished final speech

**Requires topic research to be completed first.**

## Execution Plan

For a debate topic with our side being 正方 or 反方:

1. Create todo list: `update_todo(action="create", goal="Full Preparation: [topic]")`
2. Add all tasks upfront:
   - `update_todo(action="add", task_id="research", description="Topic Research")`
   - `update_todo(action="add", task_id="speech", description="Constructive Speech")`

3. Execute Topic Research:
   - `update_todo(action="mark_in_progress", task_id="research")`
   - `run_topic_research(topic="...", side="...")`
   - `update_todo(action="mark_complete", task_id="research")`

4. Execute Constructive Speech:
   - `update_todo(action="mark_in_progress", task_id="speech")`
   - `run_constructive_speech(topic="...", side="...")`
   - `update_todo(action="mark_complete", task_id="speech")`

5. Finalize:
   - `update_todo(action="summary")`
   - Report final results to the user

## Critical Constraints

- Tasks MUST be executed in order: research → speech
- The speech tool will FAIL if research hasn't been completed
- Update todo status at each transition point
- If any step fails, report the error but continue if possible

## Output Format

After completing all tasks, provide a comprehensive summary:
- Research findings overview
- Speech quality score and iterations
- All artifact locations
- Any issues encountered

Begin preparation now.
"""
)


def get_system_prompt(mode: PrepMode) -> str:
    """Get the appropriate system prompt for the preparation mode.

    Loads the prompt from Opik via the prompt manager. Falls back to
    hardcoded prompts if the prompt manager is not initialized.

    Args:
        mode (PrepMode): Preparation mode ("lite" or "full")
    Returns:
        str: System prompt string
    """
    pm = get_prompt_manager()

    # Determine which prompt to load
    prompt_name = (
        "DEEP_PREP_ORCHESTRATOR_LITE"
        if mode == "lite"
        else "DEEP_PREP_ORCHESTRATOR_FULL"
    )

    try:
        # Load TODO instructions and mode-specific prompt from Opik
        todo_instructions = pm.get("DEEP_PREP_ORCHESTRATOR_TODO_INSTRUCTIONS")
        mode_prompt = pm.get(prompt_name)

        # Compose the prompts by chaining them together
        composed_prompt = mode_prompt + "\n\n" + todo_instructions

        logger.debug(f"Loaded and composed {prompt_name} with TODO_INSTRUCTIONS from Opik")
        return composed_prompt
    except Exception as e:
        # Fallback to hardcoded prompts
        logger.warning(
            f"Failed to load {prompt_name} from Opik: {e}. Using hardcoded fallback."
        )
        return LITE_SYSTEM_PROMPT if mode == "lite" else FULL_SYSTEM_PROMPT


# ============================================================================
# Graph Nodes
# ============================================================================


def create_call_model_node(llm_with_tools, system_prompt: str):
    """Create the model calling node with tools bound."""

    async def call_model(state: DeepPrepState, config: RunnableConfig):
        """
        Call the LLM to decide the next action.

        The LLM receives the conversation history and decides whether to:
        - Call a tool (topic_research, constructive_speech, update_todo)
        - Provide a final response to the user

        Args:
            state (DeepPrepState): Current orchestrator state
            config (RunnableConfig): Configuration for the LLM call
        Returns:
            Dict[str, List[BaseMessage]]: Updated messages with LLM response
        """
        messages = state.get("messages", [])

        # Prepend system message
        full_messages = [SystemMessage(content=system_prompt)] + list(messages)

        response = await llm_with_tools.ainvoke(full_messages, config)

        return {"messages": [response]}

    return call_model


# ============================================================================
# Graph Construction
# ============================================================================


def create_orchestrator(
    mode: PrepMode = "full",
    filesystem: Filesystem = None,
) -> StateGraph:
    """
    Create the deep preparation orchestrator graph.

    The orchestrator follows a ReAct pattern:
    1. Agent node: LLM decides which tool to call (or respond)
    2. Tool node: Executes the selected tool
    3. Loop back to agent until LLM decides to respond without tools

    Args:
        mode: Preparation mode ("lite" or "full")
        filesystem: Filesystem instance for saving artifacts.
            If None, a disk filesystem will be created.

    Returns:
        StateGraph ready for compilation

    Example:
        >>> from llm_debate_assistant.agents.deep_preparation import create_filesystem
        >>> filesystem, session_path, _ = create_filesystem("disk")
        >>> graph = create_orchestrator(mode="lite", filesystem=filesystem)
        >>> app = graph.compile()
        >>> result = await app.ainvoke(initial_state)
    """
    # Create filesystem if not provided
    if filesystem is None:
        filesystem, _, _ = create_filesystem("disk")

    # State reference for tools to access/update state
    # This is a mutable dict that will be updated during execution
    state_ref = {"state": {}}

    # Get tools for the specified mode
    tools = get_tools_for_mode(mode, filesystem, state_ref)

    # Create LLM with tools bound
    llm = get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    # Get system prompt for mode
    system_prompt = get_system_prompt(mode)

    # Create nodes
    call_model = create_call_model_node(llm_with_tools, system_prompt)
    tool_node = ToolNode(tools)

    # Build the graph
    workflow = StateGraph(DeepPrepState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges from agent
    # tools_condition routes to "tools" if tool_calls exist, otherwise END
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # After tools execute, loop back to agent
    workflow.add_edge("tools", "agent")

    return workflow


# ============================================================================
# Convenience Functions
# ============================================================================


def create_initial_state(
    topic: str,
    side: Literal["正方", "反方"],
    mode: PrepMode = "lite",
    session_id: str = None,
) -> DeepPrepState:
    """
    Create initial state for the orchestrator.

    Args:
        topic (str): The debate topic
        side (Literal["正方", "反方"]): Our side ("正方" or "反方")
        mode (PrepMode): Preparation mode ("lite" or "full")
        session_id (str, optional): Optional session ID (auto-generated if not provided)

    Returns:
        DeepPrepState: Initial DeepPrepState ready for graph invocation
    """
    if session_id is None:
        session_id = str(uuid4())

    # Create initial user message
    mode_desc = "精简" if mode == "lite" else "完整"
    initial_message = HumanMessage(
        content=f"请开始准备辩题：{topic}\n我方立场：{side}\n准备模式：{mode_desc}模式"
    )

    return DeepPrepState(
        topic=topic,
        side=side,
        session_id=session_id,
        mode=mode,
        messages=[initial_message],
        stage_status={},
        artifact_paths={},
        todo_list=None,
    )


async def run_preparation(
    topic: str,
    side: Literal["正方", "反方"],
    mode: PrepMode = "lite",
    filesystem_type: Literal["disk", "virtual"] = "disk",
) -> DeepPrepState:
    """
    Run the complete debate preparation workflow.

    This is the main entry point for running debate preparation.

    Args:
        topic (str): The debate topic
        side (Literal["正方", "反方"]): Our side ("正方" or "反方")
        mode (PrepMode): Preparation mode ("lite" or "full")
        filesystem_type (Literal["disk", "virtual"]): Type of filesystem to use

    Returns:
        DeepPrepState: Final DeepPrepState with all results

    Example:
        >>> result = await run_preparation(
        ...     topic="应该强制要求大型科技公司开源其核心算法",
        ...     side="正方",
        ...     mode="lite"
        ... )
        >>> print(result["stage_status"])
    """
    # Create filesystem
    filesystem, session_path, cache_found = create_filesystem(
        filesystem_type=filesystem_type,
        topic=topic,
        side=side,
        segment="orchestrator",
        use_cache=True,
    )

    if cache_found:
        logger.info(f"Found existing session: {session_path}")
    else:
        logger.info(f"Creating new session: {session_path}")

    # Save session metadata
    save_session_metadata(filesystem, topic, side, segment="orchestrator")

    # Create and compile the graph
    graph = create_orchestrator(mode=mode, filesystem=filesystem)
    app = graph.compile()

    # Create initial state
    initial_state = create_initial_state(
        topic=topic,
        side=side,
        mode=mode,
    )

    # Run the graph
    logger.info(f"Starting {mode} preparation for: {topic} ({side})")
    final_state = await app.ainvoke(initial_state)

    logger.info("Preparation completed")

    return final_state
