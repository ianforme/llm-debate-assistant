"""
Helper functions for agent nodes.
"""

from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.agents.deep_preparation.console import console
from llm_debate_assistant.services.llm import get_llm


def build_selective_state(state: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Build a selective state dict with only the specified keys.

    This reduces context size for LLM calls by including only relevant state.

    Args:
        state (Dict[str, Any]): Full state dictionary
        keys (List[str]): List of keys to include in selective state

    Returns:
        Dict[str, Any]: Dict containing only the specified keys from state
    """
    return {k: state.get(k) for k in keys}


async def summarize_evaluation_feedback(feedback: str, config: RunnableConfig) -> str:
    """Summarize evaluation feedback into key actionable items.

    Uses a fast LLM call to extract the most important issues and action items
    from verbose evaluation feedback.

    Args:
        feedback (str): Evaluation feedback text to summarize
        config (RunnableConfig): Configuration for the runnable context

    Returns:
        str: Concise summary of key issues and action items
    """
    summarize_prompt = """请将以下评估反馈压缩为简洁的行动要点列表。

    要求：
    - 提取3-5个最重要的改进点
    - 每点用一句话概括
    - 保留具体可操作的建议
    - 总长度控制在200-400字符

    评估反馈：
    {feedback}

    请直接输出要点列表，格式如：
    1. [要点1]
    2. [要点2]
    ..."""

    # Relative lower temperature for consistent summarization
    llm = get_llm(temperature=0.3)

    try:
        response = await llm.ainvoke(summarize_prompt.format(feedback=feedback), config)

        # Extract text from response
        if hasattr(response, "content"):
            content = response.content
            # Handle case where content is a list (Gemini format)
            if isinstance(content, list):
                summary = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            else:
                summary = str(content)
        else:
            summary = str(response)

        return summary.strip()
    except Exception as e:
        # Fallback to first 500 chars if summarization fails
        console.print(f"[dim][Warning] Feedback summarization failed: {e}[/dim]")
        return feedback[:500] + "..." if len(feedback) > 500 else feedback


def build_task_list_section(todos: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    """Build the task list section for the system prompt.

    Args:
        todos (List[Dict[str, Any]]): List of todo items
        state (Dict[str, Any]): Current state (for iteration info)

    Returns:
        str: Formatted task list section string
    """
    if not todos:
        # No todos yet - prompt agent to create them
        return """**No task plan exists yet.**

    Your first action must be to create a task plan using `write_todos_tool`.
    Based on the debate topic and side, plan what tasks you need to complete.

    Example task plan:
    ```
    [
    {"content": "破题并创建大纲 - 定义关键词、比较标准、3个论点", "status": "pending"},
    {"content": "搜集论据 - 为每个论点寻找支持性论据", "status": "pending"},
    {"content": "撰写立论稿 - 基于大纲和论据撰写开篇立论", "status": "pending"},
    {"content": "评估立论稿 - 评估质量并根据反馈改进", "status": "pending"}
    ]
    ```"""

    # Display current task list with status
    task_lines = ["**Current Task List:**\n"]
    for i, todo in enumerate(todos):
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        # Status indicators
        if status == "completed":
            indicator = "✅"
        elif status == "in_progress":
            indicator = "🔄"
        else:
            indicator = "⬜"

        task_lines.append(f"{indicator} {i}. {content}")

    # Add progress summary
    completed = sum(1 for t in todos if t.get("status") == "completed")
    in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
    pending = sum(1 for t in todos if t.get("status") == "pending")

    task_lines.append(
        f"\nProgress: {completed} completed, {in_progress} in progress, " f"{pending} pending"
    )

    # Add iteration info if in evaluation phase
    iteration_count = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)
    if state.get("evaluation"):
        task_lines.append(f"Evaluation iteration: {iteration_count}/{max_iterations}")

    return "\n".join(task_lines)
