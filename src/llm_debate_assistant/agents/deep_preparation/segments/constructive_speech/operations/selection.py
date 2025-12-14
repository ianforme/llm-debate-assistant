# -*- coding: utf-8 -*-
"""
Strategic selection operation.

Generates the strategic blueprint for constructive speech.
"""

import logging
from typing import Literal

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.schema import (
    ConstructiveStrategy,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    TopicResearchResult,
)

logger = logging.getLogger(__name__)


async def generate_constructive_strategy(
    topic: str,
    side: Literal["正方", "反方"],
    research: TopicResearchResult,
) -> ConstructiveStrategy:
    """Generate the strategic blueprint for the constructive speech.

    Analyzes research output and orchestrates:
    - Framing definitions
    - The 'Trinity' of arguments (Hook, Pivot, Anchor)
    - Narrative tone and value premise

    Args:
       topic (str): The debate topic.
       side (Literal["正方", "反方"]): Which side we are arguing for.
       research (TopicResearchResult): The topic research results.

    Returns:
      ConstructiveStrategy: The generated strategy.
    """
    logger.info(f"Generating constructive strategy for {topic} ({side})")

    # --- 1. Build Prompt Context ---

    # Key Terms
    key_terms_section = "\n".join(
        f"""
        **{i+1}. {term.term}**
        - Def: {term.strategic_definition.definition}
        - Strategy: {term.strategic_definition.inclusion_exclusion}
        - Trap to Avoid: {term.opponents_trap}
        """
        for i, term in enumerate(research.key_terms)
    )

    # Arguments (Rich Context for Selection)
    arguments_section = "\n".join(
        f"""
        **[Candidate {i+1}] {arg.claim}**
        - Type: {arg.type}
        - Warrant: {arg.warrant[:300]}...
        - Impact: {arg.impact[:200]}...
        - Evidence Snippet: {str(arg.evidence)[:150]}...
        """
        for i, arg in enumerate(research.our_research.arguments)
    )

    # Analysis Sections
    strategic_recs = "\n".join(f"- {rec}" for rec in research.analysis.strategic_recommendations)
    our_advs = "\n".join(f"- {a}" for a in research.analysis.our_advantages)
    opp_vulns = "\n".join(f"- {v}" for v in research.analysis.opponent_vulnerabilities)

    # --- 2. Call LLM ---

    pm = get_prompt_manager()
    prompt = pm.get("CONSTRUCTIVE_STRATEGY_PROMPT").format(
        topic=topic,
        side=side,
        key_terms_section=key_terms_section,
        arguments_section=arguments_section,
        strategic_recommendations=strategic_recs,
        our_advantages=our_advs,
        opponent_vulnerabilities=opp_vulns,
        value_framework=research.our_research.value_framework,
        comparison_standard=research.our_research.comparison_standard,
    )

    # Use OpenAI + JSON Mode for complex structural orchestration
    llm = get_llm(provider="openai", temperature=0.4)
    structured_llm = llm.with_structured_output(ConstructiveStrategy)

    strategy = await structured_llm.ainvoke(prompt)

    logger.info(f"Strategy Generated: Tone='{strategy.speech_tone}'")

    return strategy
