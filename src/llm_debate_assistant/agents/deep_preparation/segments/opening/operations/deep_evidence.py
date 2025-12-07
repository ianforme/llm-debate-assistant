# -*- coding: utf-8 -*-
"""
Deep evidence search operation.

Performs focused, deep evidence search for the 3 selected arguments.
Aims for 8-12 high-quality sources per argument.
"""

import json
import logging
from typing import Any

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.web_search import search_multiple_arguments
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.opening.schema import (
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

    logger.info(f"Deep evidence search for {len(strategy.selected_arguments)} arguments")

    # Prepare arguments for concurrent search
    # Format: [(argument, warrant, evidence_needed), ...]
    search_arguments = [
        (
            arg.claim,
            arg.reasoning,
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

    logger.info("Starting concurrent evidence search...")

    # Search for all arguments concurrently
    search_results = await search_multiple_arguments(
        arguments=search_arguments,
        topic=topic,
        side=side,
        model="gemini-3-pro-preview",
        use_threaded=True,  # More reliable for concurrent requests
    )

    logger.info(f"Search completed for {len(search_results)} arguments")

    # Process each result with LLM to extract structured evidence
    deep_evidence = []

    for i, (search_result, selected_arg) in enumerate(
        zip(search_results, sorted(strategy.selected_arguments, key=lambda x: x.order))
    ):
        logger.info(f"Processing evidence for argument {i+1}: {selected_arg.claim[:50]}...")

        if search_result.error:
            logger.error(f"Search error for argument {i+1}: {search_result.error}")
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

        # Extract search results
        search_text = search_result.results.get("text", "")
        search_results_list = search_result.results.get("search_results", [])

        logger.info(f"Found {len(search_results_list)} sources for argument {i+1}")

        # Use LLM to extract structured evidence from search results
        llm = get_llm(provider="openai", temperature=0.3)

        # Get prompt template from prompt manager
        pm = get_prompt_manager()
        extraction_prompt_template = pm.get("DEEP_EVIDENCE_PROMPT")

        # Format source list
        source_list = chr(10).join(
            f"{i+1}. {result.get('title', 'N/A')} - {result.get('uri', 'N/A')}"
            for i, result in enumerate(search_results_list)
        )

        # Format the prompt with actual values
        extraction_prompt = extraction_prompt_template.format(
            claim=selected_arg.claim,
            reasoning=selected_arg.reasoning,
            search_text=search_text,
            source_count=len(search_results_list),
            source_list=source_list,
        )

        # Structure extraction (without Pydantic schema to allow dynamic parsing)
        evidence_data = await llm.ainvoke(extraction_prompt)

        try:
            if hasattr(evidence_data, "content"):
                content = evidence_data.content
                # Handle both string and list content types
                if isinstance(content, list):
                    evidence_dict = json.loads(str(content))
                else:
                    evidence_dict = json.loads(content)
            else:
                evidence_dict = json.loads(str(evidence_data))
        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLM output as JSON for argument {i+1}, using fallback")
            # Fallback: create from search results directly
            evidence_dict = {
                "sources": [
                    {
                        "url": r.get("uri", ""),
                        "title": r.get("title", ""),
                        "snippet": "",
                        "credibility_note": "From search results",
                    }
                    for r in search_results_list[:10]
                ],
                "best_quotes": [],
                "statistics": [],
                "case_studies": [],
            }

        # Create ArgumentEvidence object
        arg_evidence = ArgumentEvidence(
            argument_claim=selected_arg.claim,
            sources=[EvidenceSource(**src) for src in evidence_dict.get("sources", [])[:12]],
            best_quotes=evidence_dict.get("best_quotes", [])[:5],
            statistics=evidence_dict.get("statistics", []),
            case_studies=evidence_dict.get("case_studies", []),
        )

        deep_evidence.append(arg_evidence)

        logger.info(
            f"Extracted {len(arg_evidence.sources)} sources, "
            f"{len(arg_evidence.best_quotes)} quotes for argument {i+1}"
        )

    # Save to filesystem
    evidence_json = [ev.model_dump() for ev in deep_evidence]
    filesystem.write(  # type: ignore[attr-defined]
        "/deep_evidence.json",
        json.dumps(evidence_json, indent=2, ensure_ascii=False),
    )

    logger.info(f"Deep evidence search complete for all {len(deep_evidence)} arguments")

    return {"deep_evidence": deep_evidence}
