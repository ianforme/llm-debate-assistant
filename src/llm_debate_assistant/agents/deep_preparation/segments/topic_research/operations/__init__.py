# -*- coding: utf-8 -*-
"""
Operations for topic research.

This package contains all the individual research operations.
"""

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations.define_terms import (
    define_key_terms,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations.research_side import (
    research_perspective,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations.compare import (
    comparative_analysis,
)

__all__ = [
    "define_key_terms",
    "research_perspective",
    "comparative_analysis",
]
