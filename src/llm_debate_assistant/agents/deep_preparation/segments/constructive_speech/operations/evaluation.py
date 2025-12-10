# -*- coding: utf-8 -*-
"""
Evaluation and improvement operations.

Evaluates opening statement quality and provides improvement guidance.
"""

import logging
from typing import Any


def count_visible_chars(text: str) -> int:
    """Count visible characters (excluding spaces, newlines, tabs).

    This is the standard for debate character limits - only counts
    Chinese characters, punctuation, English letters, and numbers.
    """
    return len("".join(c for c in text if not c.isspace()))


from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.schema import (
    OpeningState,
    EvaluationResult,
)
from llm_debate_assistant.agents.deep_preparation.storage import save_session_metadata

logger = logging.getLogger(__name__)


async def evaluate_statement_node(state: OpeningState) -> dict[str, Any]:
    """Evaluate opening statement quality.

    Checks:
    - Strategic alignment with selected strategy
    - Evidence usage and citation quality
    - Logical coherence and flow
    - Length compliance (max 1200 chars)
    - Oral style and persuasiveness
    """
    draft = state["draft"]
    strategy = state["opening_strategy"]
    deep_evidence = state["deep_evidence"]
    topic = state["topic"]
    side = state["side"]

    # Type narrowing assertions
    assert draft is not None
    assert strategy is not None
    assert deep_evidence is not None

    char_count = count_visible_chars(draft)
    logger.info(
        f"Evaluating opening statement ({char_count} visible characters, {len(draft)} total)"
    )

    # Build formatted sections for the prompt
    # Key terms list
    key_terms_list = ", ".join(term.term for term in strategy.selected_key_terms)

    # Arguments list
    arguments_list = chr(10).join(
        f"{i+1}. {arg.claim}"
        for i, arg in enumerate(
            sorted(strategy.selected_arguments, key=lambda x: x.order)
        )
    )

    # Evidence preview - compact summary for context
    evidence_preview = chr(10).join(
        f"论点{i+1}: {len(ev.sources)}个搜索来源, "
        f"{'有' if ev.best_quotes else '无'}搜索结果文本"
        for i, ev in enumerate(deep_evidence)
    )

    # Get prompt template from prompt manager
    pm = get_prompt_manager()
    prompt_template = pm.get("OPENING_EVALUATION_PROMPT")

    # Format the prompt with all variables
    prompt = prompt_template.format(
        topic=topic,
        side=side,
        key_terms_list=key_terms_list,
        arguments_list=arguments_list,
        evidence_preview=evidence_preview,
        value_framework=strategy.value_framework,
        comparison_standard=strategy.comparison_standard,
        rhetorical_approach=strategy.rhetorical_approach,
        draft_length=count_visible_chars(draft),
        draft=draft,
    )

    # Use OpenAI for structured output (Gemini thinking mode conflicts with structured output)
    llm = get_llm(provider="openai", temperature=0.3)  # Low temperature for consistency
    structured_llm = llm.with_structured_output(EvaluationResult)

    logger.info("Calling OpenAI LLM for evaluation...")
    evaluation = await structured_llm.ainvoke(prompt)

    # Debug logging
    logger.info(f"Structured output type: {type(evaluation)}")

    # Handle union type
    if isinstance(evaluation, dict):
        logger.info("Converting dict to EvaluationResult")
        evaluation = EvaluationResult(**evaluation)
    elif not isinstance(evaluation, EvaluationResult):
        logger.error(f"Unexpected type from Gemini: {type(evaluation)}")
        raise TypeError(
            f"Expected EvaluationResult or dict, got {type(evaluation)}. "
            f"This might be a Gemini structured output issue."
        )

    logger.info(
        f"Evaluation complete: {evaluation.result} (score: {evaluation.score}/10)"
    )

    # Increment iteration count
    new_iteration_count = state.get("iteration_count", 0) + 1

    # Save session metadata when evaluation is complete (pass or max iterations reached)
    max_iterations = state.get("max_iterations", 3)
    filesystem = state.get("filesystem")

    if filesystem and (
        evaluation.result == "pass" or new_iteration_count >= max_iterations
    ):
        logger.info("Saving opening session metadata for cache lookup")
        save_session_metadata(
            filesystem=filesystem,
            topic=state["topic"],
            side=state["side"],
            segment="opening",
        )

    return {
        "evaluation": evaluation.model_dump(),
        "iteration_count": new_iteration_count,
    }


async def improve_statement_node(state: OpeningState) -> dict[str, Any]:
    """Improve opening statement based on evaluation feedback.

    Uses evaluation feedback to generate an improved version of the draft.

    Args:
        state (OpeningState): Current state including draft and evaluation.

    Returns:
        dict[str, Any]: Updated state with improved draft.
    """
    draft = state["draft"]
    evaluation = state["evaluation"]
    topic = state["topic"]
    side = state["side"]
    filesystem = state["filesystem"]

    # Type narrowing assertions, mypy cannot infer from dict access
    assert draft is not None
    assert evaluation is not None
    assert filesystem is not None

    logger.info(f"Improving opening statement (iteration {state['iteration_count']})")

    # Build compact context for improvement
    # Keep strengths and weaknesses for context
    strengths_list = chr(10).join(f"✓ {s}" for s in evaluation.get("strengths", []))
    weaknesses_list = chr(10).join(f"✗ {w}" for w in evaluation.get("weaknesses", []))

    # Get prompt template from prompt manager
    pm = get_prompt_manager()
    prompt_template = pm.get("OPENING_IMPROVEMENT_PROMPT")

    # Simplified prompt - removed redundant context (evidence, arguments already in draft)
    prompt = prompt_template.format(
        topic=topic,
        side=side,
        draft=draft,
        evaluation_score=evaluation["score"],
        strengths_list=strengths_list,
        weaknesses_list=weaknesses_list,
        evaluation_feedback=evaluation.get("feedback", ""),
    )

    # Use Gemini for better Chinese character counting and constraint adherence
    llm = get_llm(provider="gemini", temperature=0.8)

    logger.info("Calling Gemini LLM for improvement...")
    response = await llm.ainvoke(prompt)

    # Extract improved draft
    if hasattr(response, "content"):
        content = response.content
        # Handle both string and list content types
        if isinstance(content, list):
            improved_draft = str(content)
        else:
            improved_draft = content
    else:
        improved_draft = str(response)

    improved_draft = improved_draft.strip()

    char_count = count_visible_chars(improved_draft)
    logger.info(
        f"Improvement complete: {char_count} visible characters, {len(improved_draft)} total"
    )

    # Save improved version (in /opening/ subfolder for organization)
    filesystem.write("/constructive_speech/draft.txt", improved_draft)  # type: ignore[attr-defined]
    filesystem.write(  # type: ignore[attr-defined]
        f"/constructive_speech/draft_v{state['iteration_count']}.txt",
        improved_draft,
    )  # Keep version history

    return {"draft": improved_draft}
