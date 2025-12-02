"""
Agent node for the deep preparation workflow.

This module contains the main agent_node function that decides what tools to call.
"""

import asyncio
import traceback
from typing import Any, Dict, cast

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.agents.deep_preparation.console import console
from llm_debate_assistant.agents.deep_preparation.schema import DeepPrepState
from llm_debate_assistant.agents.deep_preparation.shared.tools import ALL_TOOLS
from llm_debate_assistant.agents.deep_preparation.observability import (
    extract_token_usage,
    get_token_usage,
    update_token_usage,
    log_message_stats,
    log_token_usage,
)
from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.agents.deep_preparation.rounds.opening.agent_nodes.prompts import (
    DEEP_PREP_SYSTEM_PROMPT,
)
from llm_debate_assistant.agents.deep_preparation.rounds.opening.agent_nodes.helpers import (
    build_task_list_section,
)


async def agent_node(state: DeepPrepState, config: RunnableConfig) -> Dict[str, Any]:
    """Agent decides what to do next.

    This node calls the LLM with tools, allowing the agent to autonomously
    decide which tools to call and when.

    Args:
        state (DeepPrepState): Current agent state
        config (RunnableConfig): Runnable configuration

    Returns:
        Dict[str, Any]: Updated state with agent's message (may include tool calls)
    """
    messages = state["messages"]

    # Build fresh system prompt on EVERY call to show updated task list
    topic = state.get("topic", "未指定")
    side = state.get("side", "未指定")

    # Remove any existing SystemMessage to replace with updated one
    messages = [m for m in messages if not isinstance(m, SystemMessage)]

    # Build dynamic task list section from state todos
    todos = state.get("todos") or []
    task_list_section = build_task_list_section(todos, cast(Dict[str, Any], state))

    system_prompt = DEEP_PREP_SYSTEM_PROMPT.format(task_list_section=task_list_section)

    system_prompt += f"""

    ======================================
    Debate Context (use these exact values):
    ======================================
    Topic (辩题): "{topic}"
    Side (立场): "{side}"
    """
    # Prepend updated system prompt to messages
    messages = [SystemMessage(content=system_prompt)] + messages

    # Get LLM with tools
    llm = get_llm(temperature=0.5)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Log message stats before LLM call
    verbose = bool(state.get("verbose", False))
    log_message_stats(messages, "Agent Input", verbose=verbose)

    # Call LLM with retry logic for empty responses
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = await llm_with_tools.ainvoke(messages, config)

            # Check if response is None (Gemini API error)
            if response is None:
                console.print(
                    "[bold red][ERROR][/bold red] Gemini API returned None "
                    "(possibly context overflow or rate limit)"
                )
                if attempt < max_retries:
                    console.print(f"[yellow]Retrying ({attempt + 1}/{max_retries})...[/yellow]")
                    await asyncio.sleep(2)
                    continue
                else:
                    raise ValueError(
                        "Gemini API returned None response after retries. "
                        "Try reducing message history or wait for rate limit reset."
                    )

            # Check if we got tool calls
            has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

            # Check if response is truly empty
            content: str | list[str | dict[Any, Any]] = ""
            if hasattr(response, "text"):
                content = response.text  # type: ignore[assignment]
            elif hasattr(response, "content"):
                content = response.content  # type: ignore[assignment]

            is_empty = False
            if isinstance(content, str) and not content:
                is_empty = True
            elif isinstance(content, list) and (
                not content
                or all(
                    not block.get("text", "") if isinstance(block, dict) else not block
                    for block in content
                )
            ):
                is_empty = True

            # If both content and tool_calls are empty, retry
            if not has_tool_calls and is_empty:
                if attempt < max_retries:
                    console.print(
                        f"[bold yellow][WARNING][/bold yellow] "
                        f"LLM returned empty response! "
                        f"Retrying ({attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(1)
                    continue
                else:
                    console.print(
                        f"[bold red][ERROR][/bold red] "
                        f"LLM returned empty response after {max_retries} retries!"
                    )

            # Response is valid, break out of retry loop
            break

        except Exception as e:
            console.print(f"[bold red][ERROR][/bold red] LLM call failed: {e}")
            console.print(f"[red]{traceback.format_exc()}[/red]")
            raise

    # Log actual token usage from API response
    actual_usage = extract_token_usage(response)

    if actual_usage["total_tokens"] > 0:
        update_token_usage(actual_usage["input_tokens"], actual_usage["output_tokens"])

    log_token_usage(actual_usage, get_token_usage(), verbose=verbose)

    return {"messages": [response]}
