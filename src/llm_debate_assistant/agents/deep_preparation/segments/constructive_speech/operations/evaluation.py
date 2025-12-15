# -*- coding: utf-8 -*-
"""
Evaluation and improvement operations.

Evaluates constructive speech quality and provides improvement guidance.
"""

import logging
from typing import List, Literal

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.schema import (
    CritiqueResult,
    ConstructiveStrategy,
    ArgumentEvidence,
)
from llm_debate_assistant.agents.deep_preparation.shared.utils import (
    count_visible_chars,
)

logger = logging.getLogger(__name__)


async def critique_constructive_speech(
    topic: str,
    side: Literal["正方", "反方"],
    draft: str,
    strategy: ConstructiveStrategy,
    deep_evidence: List[ArgumentEvidence],
) -> CritiqueResult:
    """Evaluate the constructive speech draft against strategic blueprint and evidence.

    Acts as a 'Gatekeeper'. It checks:
    1. Tactical Execution: Did we use the Hook/Pivot/Anchor structure?
    2. Evidence Fidelity: Did we use the Deep Search data?
    3. Constraints: Is the length appropriate?

    Args:
        topic (str): The debate topic.
        side (Literal["正方", "反方"]): Which side we are arguing for.
        draft (str): The draft speech content.
        strategy (ConstructiveStrategy): The constructive strategy.
        deep_evidence (List[ArgumentEvidence]): Evidence for each argument.
    Returns:
        CritiqueResult: The critique result with decision and feedback.
    """
    char_count = count_visible_chars(draft)
    logger.info(f"🧐 Critiquing constructive speech ({char_count} visible chars)...")

    # Prepare Context Variables
    # Strategy Expectations: Tell the evaluator what tactical role each argument was assigned
    strategy_expectations = chr(10).join(
        f"- Arg {arg.order} (Role: {arg.role}):\n"
        f"  Target Claim: '{arg.claim}'\n"
        f"  Required Logic: {arg.warrant[:100]}..."
        for arg in sorted(strategy.selected_arguments, key=lambda x: x.order)
    )

    # Evidence Checklist: Provide specific evidence counts as checkpoints
    evidence_checklist = "无可用深度证据"
    if deep_evidence:
        evidence_checklist = chr(10).join(
            f"- For Arg {i + 1} ({ev.argument_claim[:15]}...):\n"
            f"  Expected Stats: {len(ev.statistics)} items\n"
            f"  Expected Cases: {len(ev.case_studies)} items"
            for i, ev in enumerate(deep_evidence)
        )

    # Call LLM
    pm = get_prompt_manager()
    prompt_template = pm.get("CONSTRUCTIVE_CRITIQUE_PROMPT")

    prompt = prompt_template.format(
        topic=topic,
        side=side,
        draft=draft,
        strategy_expectations=strategy_expectations,
        evidence_checklist=evidence_checklist,
        word_count=char_count,
        comparison_standard=strategy.comparison_standard,
    )

    llm = get_llm(provider="openai", temperature=0.1)
    structured_llm = llm.with_structured_output(CritiqueResult)

    logger.info("Calling OpenAI LLM for critique...")
    critique = await structured_llm.ainvoke(prompt)

    # Handle Result & Logging
    logger.info(f"Critique Decision: {critique.decision.upper()} (Score: {critique.score})")

    if critique.decision == "needs_revision":
        logger.info(f"Issues: {critique.critical_issues}")
    else:
        logger.info("Draft passed evaluation.")

    return critique
