# -*- coding: utf-8 -*-
"""
Load topic research operation.

Loads completed topic_research results from filesystem.
"""

import json
import logging
from typing import Any

from llm_debate_assistant.agents.deep_preparation.segments.opening.schema import (
    OpeningState,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    TopicResearchResult,
)
from llm_debate_assistant.agents.deep_preparation.storage import find_session_by_topic
from llm_debate_assistant.services.disk_filesystem import DiskFilesystem

logger = logging.getLogger(__name__)


def load_research_node(state: OpeningState) -> dict[str, Any]:
    """Load topic_research results.

    Primary path: Use research_context if already provided (by orchestrator).
    Fallback path: Find and load from topic_research session (for standalone use).

    Args:
        state (OpeningState): Current state of the opening segment.

    Returns:
        dict[str, Any]: Updated state with loaded research_context.

    Raises:
        FileNotFoundError: If topic_research results not found.
    """
    # Primary path: Check if research already provided
    # (e.g., when run via orchestrator)
    # The state will already have research_context populated

    if state.get("research_context") is not None:
        logger.info("Using research_context provided in state")
        return {}  # Already loaded, no changes needed

    # Fallback path: Load from topic_research session
    topic = state["topic"]
    side = state["side"]

    logger.info(f"Loading topic research for {topic} ({side})")

    # Find the topic_research session
    # Doing a lookup based on topic, side, and segment within
    # the filesystem storage
    research_session_id = find_session_by_topic(
        topic=topic,
        side=side,
        segment="topic_research",
    )

    if not research_session_id:
        error_msg = (
            f"Topic research session not found for topic='{topic}', side='{side}'.\n"
            f"Please run the topic_research segment first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Found topic_research session: {research_session_id}")

    try:
        # Create filesystem for the topic_research session
        # TODO: accommodate other storage backends if needed
        research_fs = DiskFilesystem(session_id=research_session_id)

        # Load research from topic_research filesystem
        result_data = research_fs.read("/research/research.json")

        # Check if read was successful
        if not result_data.get("success"):
            raise FileNotFoundError(result_data.get("message", "Failed to read research.json"))

        # Extract the actual JSON content
        content = result_data.get("content")
        if content is None:
            raise FileNotFoundError("No content found in research.json")

        # Parse the JSON string
        research_dict = json.loads(content)
        research_result = TopicResearchResult(**research_dict)

        logger.info(
            f"Loaded research: {len(research_result.key_terms)} key terms, "
            f"{len(research_result.our_research.arguments)} arguments"
        )

        return {"research_context": research_result}

    except FileNotFoundError:
        error_msg = (
            f"Topic research not found for topic='{topic}', side='{side}'.\n"
            f"Please run the topic_research segment first, or use the orchestrator "
            f"to run the complete preparation workflow."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        logger.error(f"Error loading research: {e}")
        raise
