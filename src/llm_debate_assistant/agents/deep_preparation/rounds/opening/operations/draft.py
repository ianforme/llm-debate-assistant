"""
Draft statement operation.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.prompts import opening_statement_prompts
from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.agents.reflection_pattern.schema import OpeningStatement
from llm_debate_assistant.agents.deep_preparation.schema import DeepPrepState
from .helpers import load_outline_from_fs, load_evidence_analysis


async def draft_statement_node_fs(state: DeepPrepState, config: RunnableConfig) -> dict[str, Any]:
    """Draft opening statement (filesystem-aware).

    Reads outline and evidence from filesystem to save tokens.

    Args:
        state (DeepPrepState): Deep preparation agent state
        config (RunnableConfig): Runnable configuration

    Returns:
        dict[str, Any]: Updated state with drafted opening statement
    """
    outline = load_outline_from_fs(state)

    # Build context with evidence from filesystem
    debate_outline_parts = ["## 关键词定义"]
    for kw in outline.get("keyword_definitions", []):
        debate_outline_parts.append(f"- {kw['keyword']}: {kw['definition']}")

    debate_outline_parts.append(f"\n## 比较标准\n{outline['comparison_standard']['standard']}")
    debate_outline_parts.append(f"理由: {outline['comparison_standard']['justification']}\n")

    debate_outline_parts.append("## 论点与证据\n")

    # Load evidence from filesystem
    for i, arg in enumerate(outline.get("arguments", []), 1):
        debate_outline_parts.append(f"\n### 论点 {i}")
        debate_outline_parts.append(f"**论点**: {arg.get('claim', '')}")
        debate_outline_parts.append(f"**论证**: {arg.get('warrant', '')}\n")

        # Load evidence from filesystem
        analysis = load_evidence_analysis(state, i)
        if analysis:
            debate_outline_parts.append("**证据**:")
            debate_outline_parts.append(f"\n{analysis}\n")

    debate_outline = "\n".join(debate_outline_parts)

    # Generate draft
    prompt = opening_statement_prompts.opening_statement_prompt(
        debate_outline=debate_outline,
        topic=state["topic"],
        side=state["side"],
    )

    # Add feedback if redoing draft
    evaluation = state.get("evaluation")
    if evaluation and evaluation.get("feedback"):
        feedback = evaluation["feedback"]
        prompt += f"\n\n【评审反馈】\n{feedback}\n\n请根据以上反馈重新撰写立论稿。"

    # Higher temp for creativity
    llm = get_llm(temperature=0.8)
    structured_llm = llm.with_structured_output(OpeningStatement)
    draft_response = await structured_llm.ainvoke(prompt)

    # Extract content from structured output (handles both dict and BaseModel)
    if isinstance(draft_response, dict):
        draft_content = draft_response.get("content", "")
    else:
        draft_content = draft_response.content  # type: ignore[attr-defined]

    return {
        "draft": draft_content,
    }
