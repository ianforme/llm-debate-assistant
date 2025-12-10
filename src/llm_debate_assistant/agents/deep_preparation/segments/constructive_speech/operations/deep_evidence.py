# -*- coding: utf-8 -*-
"""
Deep evidence search operation.

Performs focused, deep evidence search for the 3 selected arguments.
Stores raw search results without LLM extraction - the draft LLM will extract
quotes, statistics, and case studies on the fly.
"""

import json
import logging
from typing import Any

from llm_debate_assistant.services.web_search import search_multiple_arguments
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.schema import (
    OpeningState,
    ArgumentEvidence,
    EvidenceSource,
)

logger = logging.getLogger(__name__)


async def deep_evidence_node(state: OpeningState) -> dict[str, Any]:
    """Perform deep evidence search for selected arguments.
    s
        For each of the 3 selected arguments:
        - Search for 8-12 high-quality sources
        - Extract best quotes, statistics, case studies
        - Focus on credibility and specificity

        Args:
            state (OpeningState): Current state of the opening statement workflow

        Returns:
            dict[str, Any]: Dictionary containing deep evidence results

    """
    strategy = state["opening_strategy"]
    topic = state["topic"]
    side = state["side"]
    filesystem = state["filesystem"]

    # Type narrowing assertions
    assert strategy is not None
    assert filesystem is not None

    logger.info(
        f"Deep evidence search for {len(strategy.selected_arguments)} arguments"
    )

    # Prepare arguments for concurrent search
    # Format: [(argument, warrant, evidence_needed), ...]
    search_arguments = [
        (
            arg.claim,
            arg.warrant,  # Use warrant instead of reasoning
            [
                "统计数据和研究报告",
                "权威专家观点和引用",
                "真实案例和实例",
                "政府或国际组织报告",
                "学术论文或科学研究",
            ],
        )
        for arg in sorted(strategy.selected_arguments, key=lambda x: x.order)
    ]

    # Log search parameters for debugging
    logger.info(f"Starting concurrent evidence search with:")
    logger.info(f"  Topic: {topic}")
    logger.info(f"  Side: {side}")
    logger.info(f"  Model: gemini-3-pro-preview")
    logger.info(f"  Arguments to search: {len(search_arguments)}")
    for i, (claim, reasoning, _) in enumerate(search_arguments, 1):
        logger.info(f"    Arg {i} - Claim: {claim[:80]}...")
        logger.info(f"    Arg {i} - Reasoning: {reasoning[:80]}...")

    # Search for all arguments concurrently
    search_results = await search_multiple_arguments(
        arguments=search_arguments,
        topic=topic,
        side=side,
        model="gemini-3-pro-preview",
        use_threaded=True,  # More reliable for concurrent requests
    )

    logger.info(f"Search completed for {len(search_results)} arguments")

    # Log search results summary
    for i, result in enumerate(search_results, 1):
        if result.error:
            logger.error(f"  Arg {i} - ERROR: {result.error}")
        else:
            num_results = len(result.results.get("search_results", []))
            has_text = bool(result.results.get("text", ""))
            logger.info(
                f"  Arg {i} - Found {num_results} search results, has_text: {has_text}"
            )

    # Convert raw search results to evidence format (NO LLM EXTRACTION)
    # This removes the 3 LLM call bottleneck - draft LLM will extract what it needs
    sorted_args = sorted(strategy.selected_arguments, key=lambda x: x.order)
    deep_evidence = []

    for i, (search_result, selected_arg) in enumerate(zip(search_results, sorted_args), 1):
        logger.info(
            f"Processing evidence for argument {i}: {selected_arg.claim[:50]}..."
        )

        if search_result.error:
            logger.error(f"Search error for argument {i}: {search_result.error}")
            # Create empty evidence entry
            deep_evidence.append(
                ArgumentEvidence(
                    argument_claim=selected_arg.claim,
                    sources=[],
                    best_quotes=[],
                    statistics=[],
                    case_studies=[],
                )
            )
            continue

        # Extract raw search results - no LLM processing
        search_text = search_result.results.get("text", "")
        search_results_list = search_result.results.get("search_results", [])

        logger.info(f"Argument {i}: {len(search_results_list)} sources, {len(search_text)} chars of text")

        # Create sources from raw search results (just URLs and titles)
        sources = []
        for result in search_results_list[:12]:
            sources.append(
                EvidenceSource(
                    url=result.get("uri", ""),
                    title=result.get("title", ""),
                    snippet=result.get("snippet", ""),
                    credibility_note=f"Search result for: {selected_arg.claim[:50]}",
                )
            )

        # Store raw search text in best_quotes for now (draft will use this)
        # The draft LLM will extract actual quotes, statistics, etc. on the fly
        quotes = []
        if search_text:
            # Store the full search text as a "quote" for the draft to process
            quotes.append(f"[RAW SEARCH RESULTS]\n{search_text[:2000]}")

        arg_evidence = ArgumentEvidence(
            argument_claim=selected_arg.claim,
            sources=sources,
            best_quotes=quotes,
            statistics=[],  # Draft LLM will extract these
            case_studies=[],  # Draft LLM will extract these
        )

        deep_evidence.append(arg_evidence)
        logger.info(f"Argument {i}: Stored {len(sources)} sources with raw search text")

    # Save to filesystem (in /opening/ subfolder for organization)
    evidence_json = [ev.model_dump() for ev in deep_evidence]
    filesystem.write(  # type: ignore[attr-defined]
        "/constructive_speech/deep_evidence.json",
        json.dumps(evidence_json, indent=2, ensure_ascii=False),
    )

    logger.info(f"Deep evidence search complete for all {len(deep_evidence)} arguments")

    return {"deep_evidence": deep_evidence}
