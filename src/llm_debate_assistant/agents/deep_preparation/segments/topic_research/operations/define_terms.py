# -*- coding: utf-8 -*-

"""
Key terms definition operation.

Strategically defines key terms in the debate topic
to align with our debating side.
"""

from typing import List, Literal

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    KeyTerm,
    KeyTermsList,
)


async def define_key_terms(
    topic: str, our_side: Literal["正方", "反方"], config: RunnableConfig
) -> List[KeyTerm]:
    """Define key terms with strategic positioning.

    Args:
        topic (str): The debate topic
        our_side (Literal["正方", "反方"]): Which side we are arguing for
        config (RunnableConfig): Runnable configuration

    Returns:
        List[KeyTerm]: List of strategically defined key terms
    """
    llm = get_llm(temperature=0.3)  # Lower temp for consistent definitions
    structured_llm = llm.with_structured_output(KeyTermsList)

    pm = get_prompt_manager()
    prompt = pm.get("KEY_TERMS_PROMPT").format(topic=topic, side=our_side)

    result = await structured_llm.ainvoke(prompt, config)

    # Extract terms from wrapper (type: ignore for union-attr)
    if isinstance(result, KeyTermsList):
        return result.terms
    else:
        # Fallback: try to construct from dict
        return KeyTermsList(**result).terms if isinstance(result, dict) else []
