# -*- coding: utf-8 -*-
"""
Perspective research operation.

Researches arguments for one side of the debate.
"""

import asyncio
from typing import Dict, List, Literal

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.services.web_search import search_multiple_arguments
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    Argument,
    PerspectiveResearch,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations.helpers import (
    extract_logical_chain,
    assess_strength,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    CoreClaimsList,
)

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    ValueFramework,
)


async def research_perspective(
    topic: str, side: Literal["正方", "反方"], config: RunnableConfig
) -> PerspectiveResearch:
    """Research arguments for one side of the debate.

    Args:
        topic (str): The debate topic
        side (Literal["正方", "反方"]): Which side to research
        config (RunnableConfig): Runnable configuration

    Returns:
        PerspectiveResearch: Complete perspective research with arguments and evidence
    """
    # Step 1: Get core claims (3-5)
    core_claims = await _get_core_claims(topic, side, config)

    # Step 2: Develop each argument with evidence (parallel execution)
    arguments = await asyncio.gather(
        *[_develop_argument(topic, side, claim, config) for claim in core_claims]
    )

    # Step 3: Get value framework (parallel execution)
    framework = await _get_value_framework(topic, side, core_claims, config)

    return PerspectiveResearch(
        side=side,
        core_claims=core_claims,
        arguments=list(arguments),
        value_framework=framework["value_framework"],
        comparison_standard=framework["comparison_standard"],
    )


async def _get_core_claims(
    topic: str, side: Literal["正方", "反方"], config: RunnableConfig
) -> List[str]:
    """Extract core claims for one side.

    Args:
        topic (str): Debate topic
        side (Literal["正方", "反方"]): Which side to research
        config (RunnableConfig): Runnable configuration

    Returns:
        List[str]: List of core claims (3-5)
    """

    llm = get_llm(temperature=0.5)
    structured_llm = llm.with_structured_output(CoreClaimsList)

    pm = get_prompt_manager()
    prompt = pm.get("CORE_CLAIMS_PROMPT").format(topic=topic, side=side)

    result = await structured_llm.ainvoke(prompt, config)

    # Extract claims from wrapper
    if isinstance(result, CoreClaimsList):
        core_claims = result.claims
    else:
        # Fallback: try to construct from dict
        core_claims = CoreClaimsList(**result).claims if isinstance(result, dict) else []

    # Ensure we have at least 3 claims, at most 5
    if len(core_claims) < 3:
        # If too few, we'll work with what we have
        return core_claims
    elif len(core_claims) > 5:
        # If too many, take the first 5
        return core_claims[:5]

    return core_claims


async def _develop_argument(
    topic: str, side: Literal["正方", "反方"], claim: str, config: RunnableConfig
) -> Argument:
    """Develop one argument with evidence.

    Args:
        topic (str): Debate topic
        side (Literal["正方", "反方"]): Which side to research
        claim (str): The core claim to develop
        config (RunnableConfig): Runnable configuration

    Returns:
        Argument: Developed argument with reasoning, evidence, logical chain, and strength.

    Steps:
    1. Get reasoning and logical chain from LLM
    2. Search web for evidence
    3. Extract relevant evidence from search results
    4. Assess strength
    """
    # Step 1: Get reasoning
    llm = get_llm(temperature=0.7)  # Higher temp for creative reasoning

    pm = get_prompt_manager()
    reasoning_prompt = pm.get("ARGUMENT_DEVELOPMENT_PROMPT").format(
        topic=topic, side=side, claim=claim
    )

    reasoning_response = await llm.ainvoke(reasoning_prompt, config)

    # Extract reasoning text
    if hasattr(reasoning_response, "content"):
        reasoning_text = reasoning_response.content
        if isinstance(reasoning_text, list):
            # Handle Gemini format
            reasoning_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in reasoning_text
            )
        else:
            reasoning_text = str(reasoning_text)
    else:
        reasoning_text = str(reasoning_response)

    # Extract logical chain
    logical_chain = extract_logical_chain(reasoning_text)

    # Step 2: Search for evidence
    # Format: list of (argument, warrant, evidence_needed)
    arguments_for_search: List[tuple[str, str, List[str]]] = [(claim, reasoning_text[:500], [])]

    search_results = await search_multiple_arguments(
        arguments=arguments_for_search, topic=topic, side=side
    )

    # Step 3: Extract evidence from search results
    evidence = []
    if search_results and len(search_results) > 0:
        # search_results returns ArgumentEvidence objects
        search_result_dict = {
            "analysis": (
                search_results[0].analysis if hasattr(search_results[0], "analysis") else ""
            )
        }
        evidence = await _extract_evidence(claim, search_result_dict, config)

    # Step 4: Assess strength
    strength = assess_strength(evidence)

    return Argument(
        claim=claim,
        reasoning=reasoning_text,
        evidence=evidence,
        logical_chain=logical_chain,
        strength=strength,
    )


async def _extract_evidence(claim: str, search_result: Dict, config: RunnableConfig) -> List[str]:
    """Extract evidence from search results using LLM.

    Args:
        claim (str): The claim to support
        search_result (Dict): Search result containing analysis text
        config (RunnableConfig): Runnable configuration

    Returns:
        List[str]: Extracted pieces of evidence

    """
    from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
        EvidenceList,
    )

    # Format search results for prompt
    search_text = search_result.get("analysis", "")
    if not search_text:
        return []

    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(EvidenceList)

    pm = get_prompt_manager()
    prompt = pm.get("EVIDENCE_EXTRACTION_PROMPT").format(
        claim=claim,
        search_results=search_text[:2000],  # Limit to avoid token overflow
    )

    try:
        result = await structured_llm.ainvoke(prompt, config)

        # Extract evidence from wrapper
        if isinstance(result, EvidenceList):
            return result.evidence
        else:
            # Fallback: try to construct from dict
            return EvidenceList(**result).evidence if isinstance(result, dict) else []
    except Exception:
        # If extraction fails, return empty list
        return []


async def _get_value_framework(
    topic: str,
    side: Literal["正方", "反方"],
    core_claims: List[str],
    config: RunnableConfig,
) -> Dict[str, str]:
    """Get value framework and comparison standard for one side.
    Args:
        topic (str): The debate topic
        side (Literal["正方", "反方"]): The side of the debate
        core_claims (List[str]): List of core claims for the side
        config (RunnableConfig): Runnable configuration

    Returns:
        Dict[str, str]: Dictionary containing value framework and comparison standard
    """

    llm = get_llm(temperature=0.4)
    structured_llm = llm.with_structured_output(ValueFramework)

    pm = get_prompt_manager()
    prompt = pm.get("VALUE_ADVOCACY_PROMPT").format(
        topic=topic,
        side=side,
        core_claims="\n".join(f"- {claim}" for claim in core_claims),
    )

    result = await structured_llm.ainvoke(prompt, config)

    # Handle union type (dict | BaseModel)
    if isinstance(result, ValueFramework):
        return {
            "value_framework": result.value_framework,
            "comparison_standard": result.comparison_standard,
        }
    else:
        # Fallback: try to construct from dict
        if isinstance(result, dict):
            return result
        else:
            return {"value_framework": "", "comparison_standard": ""}
