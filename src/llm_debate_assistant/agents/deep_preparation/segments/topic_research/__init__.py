"""
Topic research module for deep preparation agent.

This module provides strategic research capabilities for debate topics.
"""

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.agent import (
    TopicResearchAgent,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    TopicResearchResult,
    KeyTerm,
    Argument,
    PerspectiveResearch,
    ComparativeAnalysis,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.storage import (
    save_research_to_filesystem,
    load_research_from_filesystem,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.graph import (
    create_topic_research_graph,
    create_initial_state,
    TopicResearchState,
)

__all__ = [
    "TopicResearchAgent",
    "TopicResearchResult",
    "KeyTerm",
    "Argument",
    "PerspectiveResearch",
    "ComparativeAnalysis",
    "save_research_to_filesystem",
    "load_research_from_filesystem",
    # Graph-based API (automatic, no HITL)
    "create_topic_research_graph",
    "create_initial_state",
    "TopicResearchState",
]
