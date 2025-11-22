"""
Evaluation and improvement operations.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from rich.panel import Panel

from llm_debate_assistant.prompts import opening_statement_prompts
from llm_debate_assistant.services.llm import get_llm
from ...reflection_pattern.schema import Evaluation, OpeningStatement
from ..schema import DeepPrepState
from ..console import console
from .helpers import load_outline_from_fs, load_evidence_analysis, load_current_draft


async def evaluate_statement_node_fs(
    state: DeepPrepState, config: RunnableConfig
) -> dict[str, Any]:
    """Evaluate opening statement (filesystem-aware).

    Reads outline and draft from filesystem to save tokens.

    Args:
        state (DeepPrepState): Deep preparation agent state
        config (RunnableConfig): Runnable configuration

    Returns:
        dict[str, Any]: Updated state with evaluation
    """
    outline = load_outline_from_fs(state)

    # Build compact outline for evaluation
    debate_outline_parts = ["## 关键词定义"]
    for kw in outline.get("keyword_definitions", []):
        debate_outline_parts.append(f"- {kw['keyword']}: {kw['definition']}")

    debate_outline_parts.append(
        f"\n## 比较标准\n{outline['comparison_standard']['standard']}\n"
    )

    debate_outline_parts.append("## 论点框架")
    for i, arg in enumerate(outline.get("arguments", []), 1):
        debate_outline_parts.append(f"{i}. {arg['claim']}")

    # Add evidence summary from filesystem
    debate_outline_parts.append("\n## 可用证据")
    for i in range(1, len(outline.get("arguments", [])) + 1):
        analysis = load_evidence_analysis(state, i)
        if analysis:
            # Show first 300 chars
            preview = analysis[:300]
            if len(analysis) > 300:
                preview += "..."
            debate_outline_parts.append(f"\n论点 {i}: {preview}")

    debate_outline = "\n".join(debate_outline_parts)

    # Load draft from filesystem
    draft = load_current_draft(state)

    # Create evaluation prompt
    prompt = opening_statement_prompts.opening_statement_evaluator_prompt(
        debate_outline=debate_outline,
        topic=state["topic"],
        side=state["side"],
    )
    prompt += f"\n\n【待评估的开篇立论】\n{draft}"

    # Low temp for consistency
    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(Evaluation)

    # Get current iteration
    iteration = state.get("iteration_count", 1)

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"这是第 {iteration} 次评估。"),
    ]

    evaluation_response = await structured_llm.ainvoke(messages)

    # Extract fields from structured output (handles both dict and BaseModel)
    if isinstance(evaluation_response, dict):
        eval_feedback = evaluation_response.get("feedback", "")
        eval_next_action = evaluation_response.get("next_action", "")
        eval_result = evaluation_response.get("evaluation_result", "unknown")
        eval_dict = evaluation_response
    else:
        eval_feedback = evaluation_response.feedback  # type: ignore[attr-defined]
        eval_next_action = evaluation_response.next_action  # type: ignore[attr-defined]
        eval_result = evaluation_response.evaluation_result  # type: ignore[attr-defined]
        eval_dict = evaluation_response.model_dump()  # type: ignore[attr-defined]

    # Log the evaluation feedback
    console.print(
        Panel(
            f"[bold]Feedback:[/bold]\n{eval_feedback}\n\n"
            f"[bold]Next Action:[/bold] {eval_next_action}",
            title=f"Evaluation Result (Iteration {iteration}) - {eval_result.upper()}",
            border_style="green" if eval_result == "pass" else "yellow",
            expand=False,
        )
    )

    # Set next_action from evaluation
    next_action = eval_next_action if eval_next_action else "auto"

    return {
        "evaluation": {
            **eval_dict,
            "iteration_number": iteration,
        },
        "next_action": next_action,
        "iteration_count": iteration + 1,
    }


async def improve_statement_node_fs(
    state: DeepPrepState, config: RunnableConfig
) -> dict[str, Any]:
    """Improve opening statement based on evaluation feedback (filesystem-aware).

    Reads outline and draft from filesystem to save tokens.

    Args:
        state (DeepPrepState): Deep preparation agent state
        config (RunnableConfig): Runnable configuration

    Returns:
        dict[str, Any]: Updated state with improved draft
    """
    outline = load_outline_from_fs(state)
    evaluation = state.get("evaluation")
    if not evaluation:
        raise ValueError("Cannot improve statement without evaluation")
    feedback = evaluation["feedback"]

    # Build context with all available evidence from filesystem
    debate_outline_parts = ["## 关键词定义"]
    for kw in outline.get("keyword_definitions", []):
        debate_outline_parts.append(f"- {kw['keyword']}: {kw['definition']}")

    debate_outline_parts.append(
        f"\n## 比较标准\n{outline['comparison_standard']['standard']}\n"
    )

    debate_outline_parts.append("## 论点与完整证据\n")

    # Load all evidence from filesystem
    for i, arg in enumerate(outline.get("arguments", []), 1):
        debate_outline_parts.append(f"\n### 论点 {i}")
        debate_outline_parts.append(f"**论点**: {arg['claim']}")
        debate_outline_parts.append(f"**论证**: {arg['warrant']}\n")

        analysis = load_evidence_analysis(state, i)
        if analysis:
            debate_outline_parts.append("**证据**:")
            debate_outline_parts.append(f"\n{analysis}\n")

    debate_outline = "\n".join(debate_outline_parts)

    # Load current draft from filesystem
    current_draft = load_current_draft(state)

    # Create improvement prompt
    prompt = opening_statement_prompts.opening_statement_improver_prompt(
        debate_outline=debate_outline,
        topic=state["topic"],
        side=state["side"],
    )
    prompt += f"\n\n【当前立论稿】\n{current_draft}"
    prompt += f"\n\n【评审反馈】\n{feedback}"

    # High temp for diverse improvements
    llm = get_llm(temperature=0.8)
    structured_llm = llm.with_structured_output(OpeningStatement)
    improved_response = await structured_llm.ainvoke(prompt)

    # Extract content from structured output (handles both dict and BaseModel)
    if isinstance(improved_response, dict):
        improved_content = improved_response.get("content", "")
    else:
        improved_content = improved_response.content  # type: ignore[attr-defined]

    return {
        "draft": improved_content,
    }
