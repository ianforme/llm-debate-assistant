"""
Tool handlers using registry pattern.

Each tool has a dedicated handler function that processes the tool call
and returns the result string.
"""

from typing import Any, Callable, Coroutine, Dict, cast

from langchain_core.runnables import RunnableConfig
from rich.table import Table

from .. import console
from ..schema import DeepPrepState
from ..operations import (
    create_outline_node_fs,
    search_evidence_node_fs,
    draft_statement_node_fs,
    evaluate_statement_node_fs,
    improve_statement_node_fs,
)
from ..storage import (
    save_outline_to_filesystem,
    save_evidence_to_filesystem,
    save_draft_to_filesystem,
)
from ..observability import count_tokens, log_compaction
from .helpers import build_selective_state, summarize_evaluation_feedback

# Type alias for tool handler functions
ToolHandler = Callable[
    [Dict[str, Any], Dict[str, Any], RunnableConfig, Dict[str, Any]],
    Coroutine[Any, Any, str],
]


# =============================================================================
# Core Workflow Tool Handlers
# =============================================================================


async def handle_create_outline(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
) -> str:
    """Handle create_outline_tool execution."""
    selective_state = build_selective_state(
        current_state, ["topic", "side", "filesystem"]
    )
    result = await create_outline_node_fs(cast(DeepPrepState, selective_state), config)
    state_updates.update(result)

    # Automatically save outline
    save_result = save_outline_to_filesystem(
        cast(DeepPrepState, {**current_state, **state_updates})
    )
    state_updates.update(save_result)

    outline = result.get("outline", {})
    arguments = outline.get("arguments", [])
    num_args = len(arguments)

    # Include argument claims so agent knows what to search for
    claims_list = [arg.get("claim", "") for arg in arguments]
    claims_str = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims_list)])

    return (
        f"✅ 已创建包含 {num_args} 个论点的大纲：\n{claims_str}\n\n"
        f"已保存到 /outline.json 和 /outline.md\n\n"
        f"⚠️ **立即调用** search_evidence_tool，参数：argument_claims={claims_list}"
    )


async def handle_search_evidence(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
) -> str:
    """Handle search_evidence_tool execution."""
    selective_state = build_selective_state(
        current_state, ["topic", "side", "outline", "filesystem"]
    )
    result = await search_evidence_node_fs(cast(DeepPrepState, selective_state), config)
    state_updates.update(result)

    # Automatically save evidence
    save_result = save_evidence_to_filesystem(
        cast(DeepPrepState, {**current_state, **state_updates})
    )
    state_updates.update(save_result)

    evidence_data = result.get("evidence_data", [])
    total_sources = sum(len(ev.get("search_results", [])) for ev in evidence_data)
    return (
        f"✅ 已为 {len(evidence_data)} 个论点搜集证据 "
        f"（共 {total_sources} 个来源）。已保存到 /evidence/\n\n"
        f"⚠️ **立即调用** draft_statement_tool 撰写立论稿"
    )


async def handle_draft_statement(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
) -> str:
    """Handle draft_statement_tool execution."""
    selective_state = build_selective_state(
        current_state, ["topic", "side", "outline", "evidence_data", "filesystem"]
    )
    result = await draft_statement_node_fs(cast(DeepPrepState, selective_state), config)
    state_updates.update(result)

    # Automatically save draft
    save_result = save_draft_to_filesystem(
        cast(DeepPrepState, {**current_state, **state_updates})
    )
    state_updates.update(save_result)

    draft = result.get("draft", "")
    char_count = len(draft)
    return (
        f"✅ 已撰写开篇立论稿（{char_count} 字符）。"
        f"已保存到 /current_draft.md\n\n"
        f"⚠️ **立即调用** evaluate_statement_tool 评估立论稿"
    )


async def handle_evaluate_statement(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
    verbose: bool = False,
) -> str:
    """Handle evaluate_statement_tool execution."""
    selective_state = build_selective_state(
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
    result = await evaluate_statement_node_fs(
        cast(DeepPrepState, selective_state), config
    )
    state_updates.update(result)

    evaluation = result.get("evaluation", {})
    result_status = evaluation.get("evaluation_result", "unknown")
    feedback = evaluation.get("feedback", "No feedback")
    current_iteration = evaluation.get("iteration_number", 1)
    max_iterations = current_state.get("max_iterations", 3)

    # Summarize feedback for message history
    original_length = len(feedback)
    if original_length > 500:
        if verbose:
            console.print(
                f"[dim][Compaction] Summarizing evaluation feedback "
                f"({original_length} chars)...[/dim]"
            )

        feedback_summary = await summarize_evaluation_feedback(feedback, config)
        summary_length = len(feedback_summary)

        if verbose:
            original_tokens = count_tokens(feedback)
            summary_tokens = count_tokens(feedback_summary)
            log_compaction(
                original_length,
                summary_length,
                original_tokens,
                summary_tokens,
                verbose=verbose,
            )
    else:
        feedback_summary = feedback

    # Return status with clear next action
    base_msg = (
        f"📊 评估完成（第 {current_iteration}/{max_iterations} 次）：{result_status}\n"
        f"反馈摘要：{feedback_summary}"
    )

    if result_status == "pass":
        return base_msg + "\n\n✅ 评估通过！工作流程已完成。"
    elif current_iteration >= max_iterations:
        return (
            base_msg + f"\n\n⚠️ 已达到最大迭代次数（{max_iterations}）。工作流程结束。"
        )
    else:
        return base_msg + "\n\n⚠️ **立即调用** improve_statement_tool 改进立论稿"


async def handle_improve_statement(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
) -> str:
    """Handle improve_statement_tool execution."""
    selective_state = build_selective_state(
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
    result = await improve_statement_node_fs(
        cast(DeepPrepState, selective_state), config
    )
    state_updates.update(result)

    # Automatically save improved draft
    save_result = save_draft_to_filesystem(
        cast(DeepPrepState, {**current_state, **state_updates})
    )
    state_updates.update(save_result)

    draft = result.get("draft", "")
    char_count = len(draft)

    return (
        f"🔄 已改进立论稿（{char_count} 字符）。"
        f"已保存到 /current_draft.md\n\n"
        f"⚠️ 重要提示：\n"
        f"1. 【评估与改进】步骤尚未完成！\n"
        f"2. 你必须立即调用 evaluate_statement_tool 重新评估改进后的草稿\n"
        f"3. 只有在评估通过或达到最大迭代次数时，该步骤才算完成"
    )


# =============================================================================
# Support Tool Handlers
# =============================================================================


async def handle_write_file(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
) -> str:
    """Handle write_file_tool execution."""
    path = tool_args["path"]
    content = tool_args["content"]

    filesystem = current_state.get("filesystem")
    if not filesystem:
        return "❌ 错误：文件系统未初始化"

    filesystem.write(path, content)
    return f"✅ 已将 {len(content)} 字符写入 {path}"


async def handle_read_file(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
) -> str:
    """Handle read_file_tool execution."""
    path = tool_args["path"]

    filesystem = current_state.get("filesystem")
    if not filesystem:
        return "❌ 错误：文件系统未初始化"

    result = filesystem.read(path)
    if result["success"]:
        return f"📄 文件 {path}：\n\n{result['content']}"
    else:
        error_msg = result.get("message", "Unknown error")
        suggestions = []
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            suggestions.append("文件不存在。常见文件路径：")
            suggestions.append("  - 大纲：/outline.json 或 /outline.md")
            suggestions.append(
                "  - 证据：/evidence/argument_0.json, /evidence/argument_1.json 等"
            )
            suggestions.append("  - 草稿：/current_draft.md")
            suggestions.append("使用文件系统工具检查哪些文件存在。")

        if suggestions:
            return f"❌ 读取 {path} 时出错：{error_msg}\n\n" + "\n".join(suggestions)
        else:
            return f"❌ 读取 {path} 时出错：{error_msg}"


async def handle_write_todos(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
) -> str:
    """Handle write_todos_tool execution."""
    if "todos" not in tool_args or not tool_args["todos"]:
        return (
            "⚠️ 错误：调用 `write_todos_tool` 时未提供 `todos` 列表。"
            "请提供待办事项列表。"
        )

    todos = tool_args["todos"]

    # Convert TodoItem objects to dicts if needed (Pydantic v2)
    todos_list = []
    for todo in todos:
        if hasattr(todo, "model_dump"):
            todos_list.append(todo.model_dump())
        elif hasattr(todo, "dict"):
            todos_list.append(todo.dict())
        elif isinstance(todo, dict):
            todos_list.append(todo)
        else:
            todos_list.append({"content": str(todo), "status": "pending"})

    state_updates["todos"] = todos_list

    # Calculate progress
    completed = sum(1 for t in todos_list if t.get("status") == "completed")
    in_progress = sum(1 for t in todos_list if t.get("status") == "in_progress")
    pending = sum(1 for t in todos_list if t.get("status") == "pending")

    # Create and print a rich table for todos
    table = Table(
        title="Todo List Status", show_header=True, header_style="bold magenta"
    )
    table.add_column("Status", style="dim", width=12)
    table.add_column("Task")

    status_styles = {
        "completed": "green",
        "in_progress": "yellow",
        "pending": "red",
    }

    for todo in todos_list:
        status = todo.get("status", "pending")
        style = status_styles.get(status, "white")
        content = todo.get("content", "")

        if status == "completed":
            content = f"[strike]{content}[/strike]"

        table.add_row(
            f"[{style}]{status}[/{style}]",
            content,
        )

    console.print(table)

    return (
        f"📋 已更新待办事项：{completed} 个已完成，"
        f"{in_progress} 个进行中，{pending} 个待处理"
    )


async def handle_update_todo_status(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
) -> str:
    """Handle update_todo_status_tool execution."""
    task_index = tool_args.get("task_index")
    new_status = tool_args.get("status")

    if task_index is None:
        return "⚠️ 错误：未提供 task_index 参数"
    if not new_status:
        return "⚠️ 错误：未提供 status 参数"

    todos_list = current_state.get("todos", [])
    if not todos_list:
        return "⚠️ 错误：待办事项列表为空"

    if task_index < 0 or task_index >= len(todos_list):
        return f"⚠️ 错误：索引 {task_index} 超出范围（共 {len(todos_list)} 个任务）"

    old_status = todos_list[task_index].get("status", "pending")
    todos_list[task_index]["status"] = new_status
    state_updates["todos"] = todos_list

    task_content = todos_list[task_index].get("content", "未知任务")

    console.print(
        f"[dim]  [Partial update] Task {task_index}: {old_status} → {new_status}[/dim]"
    )

    return (
        f"✅ 已更新任务 {task_index}：{task_content}\n"
        f"状态：{old_status} → {new_status}"
    )


async def handle_mark_todo_complete(
    tool_args: Dict[str, Any],
    current_state: Dict[str, Any],
    config: RunnableConfig,
    state_updates: Dict[str, Any],
) -> str:
    """Handle mark_todo_complete_tool execution."""
    task_index = tool_args.get("task_index")

    if task_index is None:
        return "⚠️ 错误：未提供 task_index 参数"

    todos_list = current_state.get("todos", [])
    if not todos_list:
        return "⚠️ 错误：待办事项列表为空"

    if task_index < 0 or task_index >= len(todos_list):
        return f"⚠️ 错误：索引 {task_index} 超出范围（共 {len(todos_list)} 个任务）"

    todos_list[task_index]["status"] = "completed"
    state_updates["todos"] = todos_list

    task_content = todos_list[task_index].get("content", "未知任务")

    console.print(f"[green]  [✅ Completed] Task {task_index}: {task_content}[/green]")

    return f"✅ 任务 {task_index} 已完成：{task_content}"


# =============================================================================
# Tool Handler Registry
# =============================================================================


TOOL_HANDLERS: Dict[str, ToolHandler] = {
    # Core workflow tools
    "create_outline_tool": handle_create_outline,
    "search_evidence_tool": handle_search_evidence,
    "draft_statement_tool": handle_draft_statement,
    "evaluate_statement_tool": handle_evaluate_statement,  # type: ignore[dict-item]
    "improve_statement_tool": handle_improve_statement,
    # Support tools
    "write_file_tool": handle_write_file,
    "read_file_tool": handle_read_file,
    "write_todos_tool": handle_write_todos,
    "update_todo_status_tool": handle_update_todo_status,
    "mark_todo_complete_tool": handle_mark_todo_complete,
}
