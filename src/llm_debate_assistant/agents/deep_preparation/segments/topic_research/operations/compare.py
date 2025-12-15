# -*- coding: utf-8 -*-
"""
Comparative analysis operation.

Compares both sides' arguments and provides strategic recommendations.
"""

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    ComparativeAnalysis,
    PerspectiveResearch,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations.helpers import (
    format_arguments_truncated,
)


async def comparative_analysis(
    our_research: PerspectiveResearch,
    opponent_research: PerspectiveResearch,
    config: RunnableConfig,
) -> ComparativeAnalysis:
    """Perform strategic comparison of both sides.

    Args:
        our_research (PerspectiveResearch): Research for our side
        opponent_research (PerspectiveResearch): Research for opponent's side
        config (RunnableConfig): Runnable configuration

    Returns:
        ComparativeAnalysis: Comparative analysis with strategic recommendations
    """
    llm = get_llm(temperature=0.5)  # Balanced temp for analysis
    structured_llm = llm.with_structured_output(ComparativeAnalysis)

    # Prepare argument summaries
    our_args_summary = format_arguments_truncated(our_research.arguments)
    opponent_args_summary = format_arguments_truncated(opponent_research.arguments)

    # Format core claims
    our_claims_str = "\n".join(f"- {claim}" for claim in our_research.core_claims)
    opponent_claims_str = "\n".join(f"- {claim}" for claim in opponent_research.core_claims)

    # Format value framework
    our_value_framework_str = (
        f"价值框架: {our_research.value_framework}\n比较标准: {our_research.comparison_standard}"
    )

    # Assuming the global prompt manager is already initialized
    pm = get_prompt_manager()
    prompt = pm.get("COMPARATIVE_ANALYSIS_PROMPT").format(
        our_side=our_research.side,
        our_claims=our_claims_str,
        our_value_framework=our_value_framework_str,
        our_evidence_summary=our_args_summary,
        opponent_side=opponent_research.side,
        opponent_claims=opponent_claims_str,
        opponent_args_summary=opponent_args_summary,
    )

    analysis = await structured_llm.ainvoke(prompt, config)

    # Handle union type (dict | BaseModel)
    if isinstance(analysis, ComparativeAnalysis):
        return analysis
    else:
        # Construct from dict
        return (
            ComparativeAnalysis(**analysis)
            if isinstance(analysis, dict)
            else ComparativeAnalysis(
                key_clashes=[],
                our_advantages=[],
                opponent_vulnerabilities=[],
                strategic_recommendations=[],
            )
        )
