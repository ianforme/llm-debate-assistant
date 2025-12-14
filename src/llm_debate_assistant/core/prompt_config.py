"""Configuration for prompts fetched from Comet Opik.

This module centralizes the naming configuration for prompts that are
fetched from the Opik platform. These prompt names are used during
initialization to load prompt templates into the PromptManager cache.
"""

from typing import List

# Prompt names to fetch from Opik
OPIK_PROMPT_NAMES: List[str] = [
    # Topic Research prompts
    "KEY_TERMS_PROMPT",
    "CORE_CLAIMS_PROMPT",
    "ARGUMENT_DEVELOPMENT_PROMPT",
    "VALUE_ADVOCACY_PROMPT",
    "COMPARATIVE_ANALYSIS_PROMPT",
    "DEEP_EVIDENCE_PROMPT",
    # Constructive Speech prompts
    "CONSTRUCTIVE_STRATEGY_PROMPT",
    "EVIDENCE_SYNTHESIS_PROMPT",
    "WRITING_CONSTRUCTIVE_SPEECH_PROMPT",
    "CONSTRUCTIVE_CRITIQUE_PROMPT",
    # Deep Preparation Orchestrator prompts
    "DEEP_PREP_ORCHESTRATOR_TODO_INSTRUCTIONS",
    "DEEP_PREP_ORCHESTRATOR_LITE",
    "DEEP_PREP_ORCHESTRATOR_FULL",
]


__all__ = ["OPIK_PROMPT_NAMES"]
