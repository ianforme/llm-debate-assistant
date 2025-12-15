# -*- coding: utf-8 -*-
"""
Perspective research operation.

Researches arguments for one side of the debate.
"""

import asyncio
from typing import Dict, List, Literal, Optional

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.services.web_search import search_queries
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    Argument,
    ArgumentDraft,
    CoreArgument,
    CoreClaimsOutput,
    KeyTerm,
    PerspectiveResearch,
    ValueFramework,
)


async def research_perspective(
    topic: str,
    side: Literal["正方", "反方"],
    key_terms: List[KeyTerm],
    config: RunnableConfig,
    opponent_research: Optional[PerspectiveResearch] = None,
) -> PerspectiveResearch:
    """Research arguments for one side of the debate with adversarial reasoning support.

    Args:
        topic (str): The debate topic
        side (Literal["正方", "反方"]): Which side to research
        key_terms (List[KeyTerm]): Strategic key term definitions from prior analysis (required)
        config (RunnableConfig): Runnable configuration
        opponent_research (Optional[PerspectiveResearch]): Opponent's research for adversarial pre-emption

    Returns:
        PerspectiveResearch: Complete perspective research with arguments and evidence
    """
    # Step 1: Prepare intelligence (if available)
    opponent_claims_text = ""
    if opponent_research and opponent_research.arguments:
        # Extract opponent's core claims as intelligence
        opponent_claims_text = "\n".join([f"- {arg.claim}" for arg in opponent_research.arguments])

    # Step 2: Get core arguments with full structure (type, claim, warrant, impact)
    # Returns List[CoreArgument] with strategic context and opponent intelligence
    core_arguments: List[CoreArgument] = await _get_core_claims(
        topic=topic,
        side=side,
        key_terms=key_terms,
        opponent_claims_text=opponent_claims_text,
        config=config,
    )

    # Step 3: Develop arguments with evidence (parallel execution)
    # Pass full CoreArgument objects to maintain logical consistency
    developed_arguments = await asyncio.gather(
        *[_develop_argument(topic, side, arg, config) for arg in core_arguments]
    )

    # Step 4: Get value framework
    # Extract claim strings for framework generation
    core_claims_str_list = [arg.claim for arg in core_arguments]
    framework = await _get_value_framework(topic, side, core_claims_str_list, config)

    return PerspectiveResearch(
        side=side,
        core_claims=core_claims_str_list,  # Store as strings for compatibility
        arguments=list(developed_arguments),
        value_framework=framework["value_framework"],
        comparison_standard=framework["comparison_standard"],
    )


async def _get_core_claims(
    topic: str,
    side: Literal["正方", "反方"],
    key_terms: List[KeyTerm],
    opponent_claims_text: str = "",
    config: RunnableConfig = None,
) -> List[CoreArgument]:
    """Extract core arguments with full structure (claim, warrant, impact) for one side.

    Args:
        topic (str): Debate topic
        side (Literal["正方", "反方"]): Which side to research
        key_terms (List[KeyTerm]): Strategic key term definitions for anchoring
        opponent_claims_text (str, optional): Intelligence on opponent's claims for pre-emption
        config (RunnableConfig): Runnable configuration

    Returns:
        List[CoreArgument]: List of core arguments with full structure (3-5)
    """
    # Format key terms definitions for the prompt
    terms_str = (
        "\n".join(
            [f"- **{term.term}**: {term.strategic_definition.definition}" for term in key_terms]
        )
        if key_terms
        else "（暂无关键定义）"
    )

    # Get LLM with higher temperature for creativity
    llm = get_llm(temperature=0.7)
    structured_llm = llm.with_structured_output(CoreClaimsOutput)

    pm = get_prompt_manager()
    prompt_str = pm.get("CORE_CLAIMS_PROMPT").format(
        topic=topic,
        side=side,
        terms=terms_str,
        opponent_claims_summary=opponent_claims_text or "（暂无情报）",
    )

    result = await structured_llm.ainvoke(prompt_str, config)

    # Extract claims from structured output
    if isinstance(result, CoreClaimsOutput):
        core_arguments = result.claims
    else:
        # Fallback: try to construct from dict
        core_arguments = CoreClaimsOutput(**result).claims if isinstance(result, dict) else []

    # Ensure we have at least 3 claims, at most 5
    if len(core_arguments) < 3:
        # If too few, we'll work with what we have
        return core_arguments
    elif len(core_arguments) > 5:
        # If too many, take the first 5
        core_arguments = core_arguments[:5]

    # Return full CoreArgument objects (preserves type, claim, warrant, impact)
    return core_arguments


async def _develop_argument(
    topic: str, side: Literal["正方", "反方"], core_arg: CoreArgument, config: RunnableConfig
) -> Argument:
    """Develop one argument using the core argument structure with lightweight validation.

    Args:
        topic (str): Debate topic
        side (Literal["正方", "反方"]): Which side to research
        core_arg (CoreArgument): Core argument with pre-generated claim, warrant, and impact
        config (RunnableConfig): Runnable configuration

    Returns:
        Argument: Fully developed argument with type, warrant, impact, and evidence.

    Steps:
    1. Get lightweight argument draft (type, reasoning, logical chain, 1-2 broad search queries)
    2. Execute quick sanity check searches (not for detailed evidence)
    3. Merge with core argument structure for final Argument object

    Note:
        The core_arg already contains claim, warrant, and impact from the strategic
        planning phase. We use these to maintain logical consistency and add minimal
        evidence for sanity checking.
    """
    # Step 1: Get lightweight argument draft for search queries
    # Pass the core warrant as context so LLM can expand on it (not fabricate from scratch)
    llm = get_llm(provider="openai", temperature=0.4)  # Lower temp for structured output
    structured_llm = llm.with_structured_output(ArgumentDraft)

    pm = get_prompt_manager()
    # Pass warrant_context to help LLM expand on the strategic reasoning
    # instead of generating completely new logic
    reasoning_prompt = pm.get("ARGUMENT_DEVELOPMENT_PROMPT").format(
        topic=topic,
        side=side,
        claim=core_arg.claim,
        warrant_context=core_arg.warrant,  # Provide strategic warrant as context
    )

    # Get structured draft with expanded reasoning and 1-2 broad queries
    draft_result: ArgumentDraft = await structured_llm.ainvoke(reasoning_prompt, config)

    print(f"Verifying claim: '{core_arg.claim[:30]}...'")

    # --- Step 2: Execute Sanity Check Search ---
    # Purpose: Only to verify the topic exists/is debated, not to gather detailed evidence
    # Since we only have 1-2 broad keywords, this is fast
    raw_search_results = await search_queries(
        queries=draft_result.search_queries,
        model="gemini-2.5-flash",  # Fast model for quick checks
    )

    # --- Step 3: Minimal Evidence Processing (for context only) ---
    # Store search results for reference, but don't rely on them for the argument
    formatted_evidence = []

    for res in raw_search_results:
        if res.error or not res.content:
            continue

        # Extract source titles
        sources_titles = ", ".join([s.title for s in res.sources])

        # Format as simple evidence entry (for sanity check reference)
        evidence_entry = (
            f"### Sanity Check: {sources_titles or 'Web Search'}\n"
            f"**Query:** {res.query}\n"
            f"**Summary:** {res.content[:200]}..."  # Truncate to 200 chars
        )
        formatted_evidence.append(evidence_entry)

    # --- Step 4: Return Final Argument ---
    # Use expanded reasoning from draft_result (which built upon core_arg.warrant context)
    # This maintains consistency while adding depth
    return Argument(
        type=core_arg.type,  # Use type from strategic planning
        claim=core_arg.claim,  # Core claim from strategic planning
        warrant=draft_result.reasoning,  # Expanded reasoning (built on strategic warrant context)
        impact=core_arg.impact,  # Impact from strategic planning
        evidence=formatted_evidence,  # Sanity check results (not detailed evidence)
    )


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

    llm = get_llm(temperature=0.6)
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
