"""
Outline creation operation.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.prompts import opening_statement_prompts
from llm_debate_assistant.services.llm import get_llm
from ...reflection_pattern.schema import DebateOutline
from ..schema import DeepPrepState


async def create_outline_node_fs(
    state: DeepPrepState, config: RunnableConfig
) -> dict[str, Any]:
    """Create debate outline (filesystem-aware).

    This version does NOT return messages to avoid overwriting ToolMessages
    in the agent workflow.

    Args:
        state (DeepPrepState): Deep preparation agent state
        config (RunnableConfig): Runnable configuration

    Returns:
        dict[str, Any]: Updated state with outline (no messages)
    """
    prompt = opening_statement_prompts.debate_outline_prompt(
        topic=state["topic"], side=state["side"]
    )

    llm = get_llm(temperature=0.7)
    structured_llm = llm.with_structured_output(DebateOutline)

    try:
        outline = await structured_llm.ainvoke(prompt)

        if outline is None:
            raise ValueError("LLM returned None - API call may have failed")

        return {
            "outline": outline.model_dump(),
        }
    except Exception as e:
        raise RuntimeError(
            f"Failed to create outline: {e}\n" f"Topic: {state['topic']}"
        ) from e
