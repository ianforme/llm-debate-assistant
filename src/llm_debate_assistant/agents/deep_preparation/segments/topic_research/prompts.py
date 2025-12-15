# -*- coding: utf-8 -*-
"""
Topic research prompts are managed in Opik.

All prompts for topic research operations are stored in the Opik platform
for centralized management and version control. They are loaded at runtime
via the PromptManager service.

To initialize prompts from Opik:
    >>> import opik
    >>> from llm_debate_assistant.services.prompt_manager import get_prompt_manager
    >>>
    >>> client = opik.Opik()
    >>> pm = get_prompt_manager()
    >>> await pm.init(client)

Prompt names:
    - KEY_TERMS_PROMPT
    - CORE_CLAIMS_PROMPT
    - ARGUMENT_DEVELOPMENT_PROMPT
    - VALUE_ADVOCACY_PROMPT
    - COMPARATIVE_ANALYSIS_PROMPT
    - EVIDENCE_EXTRACTION_PROMPT
"""
