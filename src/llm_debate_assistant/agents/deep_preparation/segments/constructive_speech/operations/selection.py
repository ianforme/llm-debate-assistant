# -*- coding: utf-8 -*-
"""
Strategic selection operation.

Selects key terms and arguments from topic_research for opening statement.
"""

import logging
from typing import Any

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.schema import (
    OpeningState,
    OpeningStrategy,
)

logger = logging.getLogger(__name__)


async def select_strategy_node(state: OpeningState) -> dict[str, Any]:
    """Select strategic key terms and arguments for opening.

    Analyzes topic_research output and selects:
    - 2-3 most favorable key term definitions
    - 3 strongest arguments (considering strength, strategic fit, opponent vulnerabilities)
    - Value framework and comparison standard
    - Rhetorical approach

    Args:
       state (OpeningState): Current state including research_context.

    Returns:
      dict[str, Any]: Updated state with selected opening strategy.
    """
    research = state["research_context"]
    topic = state["topic"]
    side = state["side"]
    filesystem = state["filesystem"]

    # Type narrowing assertions
    assert research is not None
    assert filesystem is not None

    logger.info(f"Selecting opening strategy for {topic} ({side})")

    # Build formatted sections for the prompt
    # Key terms section
    key_terms_section = chr(10).join(
        f"""
      **{i+1}. {term.term}**
      - 基准定义：{term.standard_definition}
      - 我方战略定义：{term.strategic_definition.definition}
      - 权威锚点：{term.strategic_definition.authority_anchor}
      - 收纳与切割：{term.strategic_definition.inclusion_exclusion}
      - 对方定义的陷阱：{term.opponents_trap}
      - 举证责任转移：{term.burden_shift}
      """
        for i, term in enumerate(research.key_terms)
    )

    # Arguments section
    arguments_section = chr(10).join(
        f"""
      **论点{i+1}：{arg.claim}**
      - 类型：{arg.type} ({"价值论证" if arg.type == "Value" else "实利论证"})
      - 推导逻辑：{arg.warrant[:200]}...
      - 影响/实利：{arg.impact[:150]}...
      """
        for i, arg in enumerate(research.our_research.arguments)
    )

    # Strategic recommendations section
    strategic_recommendations = chr(10).join(
        f"- {rec}" for rec in research.analysis.strategic_recommendations
    )

    # Our advantages section
    our_advantages = chr(10).join(f"- {adv}" for adv in research.analysis.our_advantages)

    # Opponent vulnerabilities section
    opponent_vulnerabilities = chr(10).join(
        f"- {vuln}" for vuln in research.analysis.opponent_vulnerabilities
    )

    # Get prompt template from prompt manager
    pm = get_prompt_manager()
    prompt_template = pm.get("OPENING_STRATEGY_SELECTION_PROMPT")

    # Format the prompt with all variables
    prompt = prompt_template.format(
        topic=topic,
        side=side,
        key_terms_count=len(research.key_terms),
        key_terms_section=key_terms_section,
        arguments_count=len(research.our_research.arguments),
        arguments_section=arguments_section,
        value_framework=research.our_research.value_framework,
        comparison_standard=research.our_research.comparison_standard,
        strategic_recommendations=strategic_recommendations,
        our_advantages=our_advantages,
        opponent_vulnerabilities=opponent_vulnerabilities,
    )

    # Use OpenAI for structured output (Gemini thinking mode conflicts with structured output)
    llm = get_llm(provider="openai", temperature=0.4)
    structured_llm = llm.with_structured_output(OpeningStrategy)

    logger.info("Calling OpenAI LLM for strategic selection...")
    strategy = await structured_llm.ainvoke(prompt)

    # Debug logging to see what Gemini returns
    logger.info(f"Structured output type: {type(strategy)}")
    logger.debug(f"Structured output content: {strategy}")

    # Handle union type (dict or Pydantic model)
    if isinstance(strategy, dict):
        logger.info("Converting dict to OpeningStrategy")
        strategy = OpeningStrategy(**strategy)
    elif not isinstance(strategy, OpeningStrategy):
        # Gemini might return something unexpected - try to handle it
        logger.error(f"Unexpected type from Gemini: {type(strategy)}")
        logger.error(f"Content: {strategy}")
        raise TypeError(
            f"Expected OpeningStrategy or dict, got {type(strategy)}. "
            f"This might be a Gemini structured output issue."
        )

    logger.info(
        f"Selected strategy: {len(strategy.selected_key_terms)} terms, "
        f"3 arguments, {strategy.rhetorical_approach} approach"
    )

    # Save to filesystem (in /opening/ subfolder for organization)
    filesystem.write(  # type: ignore[attr-defined]
        "/constructive_speech/opening_strategy.json", strategy.model_dump_json(indent=2, ensure_ascii=False)
    )

    return {"opening_strategy": strategy}
