"""
Segments module for deep preparation.

This module contains all debate preparation segments that can be mixed and matched
based on tournament rules and formats.

Available segments:
- topic_research: Strategic research on debate topics and both positions
- opponent_scouting: Intelligence gathering on specific opponent teams
- opening: Opening statement preparation
- free_debate: Free debate round preparation
- rebuttal: Rebuttal round preparation
- closing: Closing statement preparation
"""

from llm_debate_assistant.agents.deep_preparation.segments.topic_research import (
    TopicResearchAgent,
)

__all__ = [
    "TopicResearchAgent",
]
