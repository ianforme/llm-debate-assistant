# -*- coding: utf-8 -*-
"""
Load topic research operation.

Loads completed topic_research results from filesystem.
"""

import logging
from typing import Literal, Optional

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    TopicResearchResult,
)
from llm_debate_assistant.agents.deep_preparation.storage import (
    DiskFilesystem,
    find_session_by_topic,
)

logger = logging.getLogger(__name__)


def load_research(
    topic: str,
    side: Literal["正方", "反方"],
    research_context: Optional[TopicResearchResult] = None,
) -> TopicResearchResult:
    """Load topic_research results with robust validation.

    Args:
        topic (str): The debate topic.
        side (Literal["正方", "反方"]): Which side we are arguing for.
        research_context (Optional[TopicResearchResult]): Optional pre-loaded research context (Orchestrator mode).

    Returns:
        TopicResearchResult: The loaded research result.

    Raises:
        FileNotFoundError: If topic_research results not found.
        RuntimeError: If research data is corrupted or invalid.
    """
    # 1. Primary path: Orchestrator injection
    if research_context is not None:
        logger.info("Using research_context provided (Orchestrator Mode)")
        return research_context

    # 2. Fallback path: Standalone Mode (Load from disk)
    logger.info(
        f"Standalone Mode: Searching for topic_research for '{topic}' ({side})..."
    )

    research_session_id = find_session_by_topic(
        topic=topic,
        side=side,
        segment="topic_research",
    )

    if not research_session_id:
        error_msg = (
            f"❌ Missing Prerequisite: Topic Research not found.\n"
            f"  Topic: {topic}\n"
            f"  Side: {side}\n"
            f"Action: Please run the 'topic_research' subgraph first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        research_fs = DiskFilesystem(session_id=research_session_id)
        result_data = research_fs.read("/research/research.json")

        if not result_data.get("success"):
            raise FileNotFoundError(f"Read failed: {result_data.get('message')}")

        content = result_data.get("content")
        if not content:
            raise ValueError("research.json is empty")

        try:
            research_result = TopicResearchResult.model_validate_json(content)
        except AttributeError:
            research_result = TopicResearchResult.parse_raw(content)

        logger.info(
            f"✅ Successfully loaded research context:\n"
            f"  - Terms: {len(research_result.key_terms)}\n"
            f"  - Our Args: {len(research_result.our_research.arguments)}\n"
            f"  - Opponent Args: {len(research_result.opponent_research.arguments)}"
        )

        return research_result

    except Exception as e:
        logger.exception(f"Failed to load research session {research_session_id}")
        raise RuntimeError(f"Corrupted research data: {str(e)}") from e
