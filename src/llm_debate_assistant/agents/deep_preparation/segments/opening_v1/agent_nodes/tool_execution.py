"""
Tool execution node for the deep preparation workflow.

This module handles executing tools requested by the agent and includes
the automatic improvement loop logic.
"""

import traceback
from typing import Any, Dict, cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.agents.deep_preparation.console import console
from llm_debate_assistant.agents.deep_preparation.schema import DeepPrepState
from llm_debate_assistant.agents.deep_preparation.segments.opening.operations import (  # type: ignore[attr-defined]
    evaluate_statement_node_fs,  # type: ignore[attr-defined]
    improve_statement_node_fs,  # type: ignore[attr-defined]
)
from llm_debate_assistant.agents.deep_preparation.storage import (
    save_draft_to_filesystem,
)
from llm_debate_assistant.agents.deep_preparation.observability import log_tool_output
from llm_debate_assistant.agents.deep_preparation.segments.opening.agent_nodes.helpers import (  # type: ignore[import-untyped]
    build_selective_state,
)
from llm_debate_assistant.agents.deep_preparation.segments.opening.agent_nodes.tool_handlers import (  # type: ignore[import-untyped]
    TOOL_HANDLERS,
)


async def tool_execution_node(state: DeepPrepState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute the tools requested by the agent.

    This node inspects the last message for tool calls and executes them,
    calling the actual implementation functions and managing state updates.

    Includes automatic routing for evaluation→improve loop to prevent
    agent confusion and ensure reliable improvement iterations.

    Args:
        state (DeepPrepState): Current agent state
        config (RunnableConfig): Runnable configuration

    Returns:
        Dict[str, Any]: Updated state with tool results
    """
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        raise ValueError("Expected AIMessage with tool_calls")

    tool_messages: list[ToolMessage] = []
    state_updates: Dict[str, Any] = {}
    tool_call_ids = [tc["id"] for tc in last_message.tool_calls]
    verbose = bool(state.get("verbose", False))

    # Execute each tool call
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        try:
            result = await _execute_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                state=state,
                config=config,
                state_updates=state_updates,
                verbose=verbose,
            )

            tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            tool_messages.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_id,
                    name=tool_name,
                    status="error",
                )
            )

    # Run automatic improvement loop if evaluation failed
    await _run_improvement_loop(
        last_message=last_message,
        state=state,
        state_updates=state_updates,
        config=config,
    )

    # Check for missing responses
    response_ids = [tm.tool_call_id for tm in tool_messages]
    missing_ids = set(tool_call_ids) - set(response_ids)
    if missing_ids:
        console.print(
            f"[bold red][tool_execution] WARNING: "
            f"Missing responses for tool_call_ids: {missing_ids}[/bold red]"
        )

    # Ensure state_updates doesn't override tool_messages
    if "messages" in state_updates:
        del state_updates["messages"]

    log_tool_output(tool_messages, verbose=verbose)

    return {
        **state_updates,
        "messages": tool_messages,
    }


async def _execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    state: DeepPrepState,
    config: RunnableConfig,
    state_updates: Dict[str, Any],
    verbose: bool = False,
) -> str:
    """Execute a single tool and return result.

    Args:
        tool_name (str): Name of the tool to execute
        tool_args (Dict[str, Any]): Arguments for the tool
        state (DeepPrepState): Current agent state
        config (RunnableConfig): Runnable configuration
        state_updates (Dict[str, Any]): Dict to accumulate state updates
        verbose (bool): Whether to log verbose output

    Returns:
        str: Tool execution result as string
    """
    # Merge any previous state updates into current state
    current_state = {**state, **state_updates}

    # Look up handler in registry
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise ValueError(f"未知工具：{tool_name}")

    # All handlers use the same signature
    return await handler(tool_args, current_state, config, state_updates)


async def _run_improvement_loop(
    last_message: AIMessage,
    state: DeepPrepState,
    state_updates: Dict[str, Any],
    config: RunnableConfig,
) -> None:
    """Run automatic improvement loop after evaluation fails.

    This prevents agent confusion and ensures reliable iterations.

    Args:
        last_message (AIMessage): Last AI message with tool calls
        state (DeepPrepState): Current agent state
        state_updates (Dict[str, Any]): Dict to accumulate state updates
        config (RunnableConfig): Runnable configuration
    """
    # Check if evaluate_statement_tool was called
    evaluation_executed = any(
        tc["name"] == "evaluate_statement_tool" for tc in last_message.tool_calls
    )

    if not evaluation_executed:
        return

    # Merge state updates to get current state
    current_state = {**state, **state_updates}
    evaluation = current_state.get("evaluation", {})
    result_status = evaluation.get("evaluation_result")
    current_iteration = evaluation.get("iteration_number", 1)
    max_iterations = current_state.get("max_iterations", 3)

    # Check if we need to auto-improve
    should_improve = result_status == "fail" and current_iteration < max_iterations

    if not should_improve:
        return

    console.print(
        f"[bold yellow]🔄 Auto-routing to improvement loop "
        f"(completed {current_iteration}/{max_iterations})[/bold yellow]"
    )

    try:
        # Step 1: Improve statement
        console.print("[dim]→ Calling improve_statement_tool...[/dim]")
        improve_selective_state = build_selective_state(
            current_state,
            [
                "topic",
                "side",
                "outline",
                "draft",
                "evaluation",
                "iteration_count",
                "max_iterations",
                "filesystem",
            ],
        )
        improve_result = await improve_statement_node_fs(
            cast(DeepPrepState, improve_selective_state), config
        )
        state_updates.update(improve_result)

        # Save improved draft
        merged_state = {**current_state, **state_updates}
        save_result = save_draft_to_filesystem(cast(DeepPrepState, merged_state))
        state_updates.update(save_result)

        draft = improve_result.get("draft", "")
        char_count = len(draft)
        console.print(f"[dim]  🔄 Auto-improved draft ({char_count} chars)[/dim]")

        # Update current state with improvements
        current_state = {**current_state, **state_updates}

        # Step 2: Re-evaluate
        console.print("[dim]→ Calling evaluate_statement_tool...[/dim]")
        eval_selective_state = build_selective_state(
            current_state,
            [
                "topic",
                "side",
                "outline",
                "draft",
                "iteration_count",
                "max_iterations",
                "filesystem",
            ],
        )
        eval_result = await evaluate_statement_node_fs(
            cast(DeepPrepState, eval_selective_state), config
        )
        state_updates.update(eval_result)

        new_evaluation = eval_result.get("evaluation", {})
        new_result_status = new_evaluation.get("evaluation_result", "unknown")
        new_iteration = new_evaluation.get("iteration_number", current_iteration + 1)
        console.print(
            f"[dim]  📊 Auto-evaluation (iteration {new_iteration}): " f"{new_result_status}[/dim]"
        )

        # Log completion status
        if new_result_status == "pass":
            console.print("[bold green]✅ 评估通过！【评估与改进】步骤已完成。[/bold green]")
        elif new_iteration >= max_iterations:
            console.print(
                f"[bold yellow]⚠️ 已达到最大迭代次数（{max_iterations}）。\n"
                f"【评估与改进】步骤标记为完成"
                f"（由于达到最大迭代次数限制）。[/bold yellow]"
            )
        else:
            console.print("[yellow]评估仍未通过，将在下一轮继续改进...[/yellow]")

    except Exception as e:
        console.print(f"[bold red]自动改进循环出错：{e}[/bold red]")
        console.print(f"[red]{traceback.format_exc()}[/red]")
