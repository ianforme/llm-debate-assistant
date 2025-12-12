# -*- coding: utf-8 -*-
"""
Deep evidence search operation.

Performs focused, deep evidence search for the 3 selected arguments
using a Plan-Execute-Synthesize workflow.
"""

import logging
from typing import Dict, List, Literal

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.services.web_search import search_queries
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.schema import (
    ArgumentEvidence,
    EvidenceSource,
    SearchPlan,
    SynthesizedEvidence,
    SelectedArgument,
)

logger = logging.getLogger(__name__)


async def deep_evidence_search(
    topic: str,
    side: Literal["正方", "反方"],
    selected_arguments: List[SelectedArgument],
    config: RunnableConfig,
) -> List[ArgumentEvidence]:
    """Perform deep evidence search for selected arguments via Plan-Execute-Synthesize loop.

    Args:
        topic (str): The debate topic.
        side (Literal["正方", "反方"]): Which side we are arguing for.
        selected_arguments (List[SelectedArgument]): The arguments to find evidence for.
        config (RunnableConfig): Runnable configuration.

    Returns:
        List[ArgumentEvidence]: Evidence for each argument.
    """
    logger.info(
        f"🔍 Starting Deep Evidence Search for {len(selected_arguments)} arguments..."
    )

    llm = get_llm(provider="openai", temperature=0.1)
    pm = get_prompt_manager()

    final_evidence_list: List[ArgumentEvidence] = []

    for i, arg in enumerate(selected_arguments):
        logger.info(f"--- Processing Arg {i+1}: {arg.claim[:30]}... ---")

        # ---------------------------------------------------------
        # Step 1: Planning
        # ---------------------------------------------------------
        plan_prompt = pm.get("DEEP_EVIDENCE_PROMPT").format(
            topic=topic,
            side=side,
            claim=arg.claim,
            warrant=arg.warrant,
            evidence_summary=arg.evidence_summary,
        )

        planner = llm.with_structured_output(SearchPlan)
        search_plan = await planner.ainvoke(plan_prompt, config)

        logger.info(f"  Queries: {search_plan.queries}")

        # ---------------------------------------------------------
        # Step 2: Execution (Web Search)
        # ---------------------------------------------------------
        raw_results = await search_queries(
            search_plan.queries, model="gemini-2.5-flash"
        )

        source_map: Dict[int, EvidenceSource] = {}
        formatted_context = ""
        valid_result_count = 0

        for idx, res in enumerate(raw_results, 1):
            if res.error or not res.content:
                continue

            primary_source = res.sources[0] if res.sources else None
            title = primary_source.title if primary_source else "Web Search Result"
            url = primary_source.uri if primary_source else ""

            source_map[idx] = EvidenceSource(
                url=url,
                title=title,
                snippet="",
                credibility_note="Deep Search Result",
            )

            formatted_context += f"""
            【Source {idx}】
            Title: {title}
            URL: {url}
            Content: {res.content[:2500]}
            ------------------------------------------------
            """
            valid_result_count += 1

        if valid_result_count == 0:
            logger.warning(f"  No valid search results found for Arg {i+1}")
            final_evidence_list.append(_create_empty_evidence(arg.claim))
            continue

        # ---------------------------------------------------------
        # Step 3: Synthesis
        # ---------------------------------------------------------
        synthesis_prompt = pm.get("EVIDENCE_SYNTHESIS_PROMPT").format(
            claim=arg.claim,
            warrant=arg.warrant,
            formatted_search_results=formatted_context,
        )

        synthesizer = llm.with_structured_output(SynthesizedEvidence)

        try:
            cleaned_data = await synthesizer.ainvoke(synthesis_prompt, config)
        except Exception as e:
            logger.error(f"Synthesis failed for Arg {i+1}: {e}")
            final_evidence_list.append(_create_empty_evidence(arg.claim))
            continue

        # ---------------------------------------------------------
        # Step 4: Assembly
        # ---------------------------------------------------------
        stats = []
        quotes = []
        cases = []
        used_sources_ids = set()

        def process_items(items, target_list):
            """Helper to process ExtractedItems and add to target list."""
            for item in items:
                source_meta = source_map.get(item.source_id)
                source_title = source_meta.title if source_meta else "Unknown"
                formatted_text = f"{item.text} (Source: {source_title})"
                target_list.append(formatted_text)
                if item.source_id in source_map:
                    used_sources_ids.add(item.source_id)

        process_items(cleaned_data.statistics, stats)
        process_items(cleaned_data.quotes, quotes)
        process_items(cleaned_data.case_studies, cases)

        final_sources = [source_map[sid] for sid in sorted(used_sources_ids)]

        logger.info(
            f"  Extracted: {len(stats)} stats, {len(quotes)} quotes, {len(cases)} cases"
        )

        final_evidence_list.append(
            ArgumentEvidence(
                argument_claim=arg.claim,
                sources=final_sources,
                best_quotes=quotes[:5],
                statistics=stats[:5],
                case_studies=cases[:5],
            )
        )

    return final_evidence_list


def _create_empty_evidence(claim: str) -> ArgumentEvidence:
    """Helper to create an empty evidence object.
    Args:
        claim (str): The argument claim.

    Returns:
        ArgumentEvidence: An empty evidence object.
    """
    return ArgumentEvidence(
        argument_claim=claim, sources=[], best_quotes=[], statistics=[], case_studies=[]
    )
