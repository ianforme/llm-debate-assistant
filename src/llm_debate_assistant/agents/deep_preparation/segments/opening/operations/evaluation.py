# -*- coding: utf-8 -*-
"""
Evaluation and improvement operations.

Evaluates opening statement quality and provides improvement guidance.
"""

import logging
from typing import Any

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.opening.schema import (
    OpeningState,
    EvaluationResult,
)

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
    topic = state["topic"]
    side = state["side"]

    # Type narrowing assertions
    assert draft is not None
    assert strategy is not None

    logger.info(f"Evaluating opening statement ({len(draft)} characters)")

    # Build formatted sections for the prompt
    # Key terms list
    key_terms_list = ", ".join(term.term for term in strategy.selected_key_terms)

    # Arguments list
    arguments_list = chr(10).join(
        f"{i+1}. {arg.claim}"
        for i, arg in enumerate(sorted(strategy.selected_arguments, key=lambda x: x.order))
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
        value_framework=strategy.value_framework,
        comparison_standard=strategy.comparison_standard,
        rhetorical_approach=strategy.rhetorical_approach,
        draft_length=len(draft),
        draft=draft,
    )

    llm = get_llm(provider="openai", temperature=0.3)  # Low temperature for consistency
    structured_llm = llm.with_structured_output(EvaluationResult)

    logger.info("Calling LLM for evaluation...")
    evaluation = await structured_llm.ainvoke(prompt)

    # Handle union type
    if isinstance(evaluation, dict):
        evaluation = EvaluationResult(**evaluation)

    # Type narrowing - ensure we have EvaluationResult
    assert isinstance(evaluation, EvaluationResult)

    logger.info(f"Evaluation complete: {evaluation.result} (score: {evaluation.score}/10)")

    # Increment iteration count
    new_iteration_count = state.get("iteration_count", 0) + 1

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
    strategy = state["opening_strategy"]
    deep_evidence = state["deep_evidence"]
    topic = state["topic"]
    side = state["side"]
    filesystem = state["filesystem"]

    # Type narrowing assertions, mypy cannot infer from dict access
    assert draft is not None
    assert evaluation is not None
    assert strategy is not None
    assert deep_evidence is not None
    assert filesystem is not None

    logger.info(f"Improving opening statement (iteration {state['iteration_count']})")

    # Build formatted sections for the prompt
    # Strengths list
    strengths_list = chr(10).join(f"✓ {s}" for s in evaluation.get("strengths", []))

    # Weaknesses list
    weaknesses_list = chr(10).join(f"✗ {w}" for w in evaluation.get("weaknesses", []))

    # Arguments list
    arguments_list = chr(10).join(
        f"{i+1}. {arg.claim}"
        for i, arg in enumerate(sorted(strategy.selected_arguments, key=lambda x: x.order))
    )

    # Deep evidence section
    deep_evidence_section = chr(10).join(
        f"""
        论点{i+1}证据：
        - 最佳引用：{'; '.join(ev.best_quotes[:3])}
        - 统计数据：{'; '.join(ev.statistics[:3]) if ev.statistics else '无'}
        - 案例：{'; '.join(ev.case_studies[:2]) if ev.case_studies else '无'}
        """
        for i, ev in enumerate(deep_evidence)
    )

    # Get prompt template from prompt manager
    pm = get_prompt_manager()
    prompt_template = pm.get("OPENING_IMPROVEMENT_PROMPT")

    # Format the prompt with all variables
    prompt = prompt_template.format(
        topic=topic,
        side=side,
        draft=draft,
        score=evaluation["score"],
        result=evaluation.get("result", "fail").upper(),
        strengths_list=strengths_list,
        weaknesses_list=weaknesses_list,
        feedback=evaluation.get("feedback", ""),
        strategy_alignment="是" if evaluation.get("strategy_alignment") else "否",
        arguments_list=arguments_list,
        deep_evidence_section=deep_evidence_section,
    )

    llm = get_llm(provider="openai", temperature=0.8)

    logger.info("Calling LLM for improvement...")
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

    logger.info(f"Improvement complete: {len(improved_draft)} characters")

    # Save improved version
    filesystem.write("/draft.txt", improved_draft)  # type: ignore[attr-defined]
    filesystem.write(  # type: ignore[attr-defined]
        f"/draft_v{state['iteration_count']}.txt",
        improved_draft,
    )  # Keep version history

    return {"draft": improved_draft}
