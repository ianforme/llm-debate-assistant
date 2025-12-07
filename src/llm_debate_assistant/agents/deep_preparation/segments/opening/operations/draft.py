# -*- coding: utf-8 -*-
"""
Draft opening statement operation.

Creates the opening statement based on selected strategy and deep evidence.
"""

import logging
from typing import Any

from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.opening.schema import (
    OpeningState,
)

logger = logging.getLogger(__name__)


async def draft_statement_node(state: OpeningState) -> dict[str, Any]:
    """Draft opening statement based on strategy and evidence.

    Creates a 4-minute opening statement (max 1200 Chinese characters):
    - Oral/conversational style
    - Clear structure: definitions → arguments → conclusion
    - Evidence integration
    - Aligned with selected strategy

    Args:
        state (OpeningState): Current state of the opening statement workflow

    Returns:
        dict[str, Any]: Dictionary containing the drafted opening statement
    """
    strategy = state["opening_strategy"]
    deep_evidence = state["deep_evidence"]
    topic = state["topic"]
    side = state["side"]
    filesystem = state["filesystem"]

    # Type narrowing assertions
    assert strategy is not None
    assert deep_evidence is not None
    assert filesystem is not None

    logger.info(f"Drafting opening statement for {topic} ({side})")

    # Get evaluation feedback if this is an improvement iteration
    evaluation = state.get("evaluation")
    improvement_feedback = ""
    if evaluation and evaluation.get("result") == "fail":
        improvement_feedback = f"""
        【上一版本评审反馈】
        {evaluation.get('feedback', '')}

        请根据反馈改进本次草稿。
        """

    # Build formatted sections for the prompt
    # Key terms section
    key_terms_section = chr(10).join(
        f"""
        **{i+1}. {term.term}**
        我方定义：{term.our_definition}
        选择理由：{term.rationale}
        """
        for i, term in enumerate(strategy.selected_key_terms)
    )

    # Arguments section
    arguments_section = chr(10).join(
        f"""
        **论点{arg.order}：{arg.claim}**
        推理：{arg.reasoning}
        逻辑链：{arg.logical_chain}
        选择理由：{arg.selection_rationale}
        """
        for arg in sorted(strategy.selected_arguments, key=lambda x: x.order)
    )

    # Deep evidence section
    deep_evidence_section = chr(10).join(
        f"""
        **论点{i+1}证据：{ev.argument_claim}**

        来源数量：{len(ev.sources)}个高质量来源

        最佳引用：
        {chr(10).join(f"- {quote}" for quote in ev.best_quotes)}

        统计数据：
        {chr(10).join(f"- {stat}" for stat in ev.statistics) if ev.statistics else "（无）"}

        案例研究：
        {chr(10).join(f"- {case}" for case in ev.case_studies) if ev.case_studies else "（无）"}

        来源列表：
        {chr(10).join(f"{j+1}. {src.title} ({src.credibility_note})" for j, src in enumerate(ev.sources[:5]))}
        """
        for i, ev in enumerate(deep_evidence)
    )

    # Get prompt template from prompt manager
    pm = get_prompt_manager()
    prompt_template = pm.get("OPENING_DRAFT_PROMPT")

    # Format the prompt with all variables
    prompt = prompt_template.format(
        topic=topic,
        side=side,
        key_terms_count=len(strategy.selected_key_terms),
        key_terms_section=key_terms_section,
        arguments_section=arguments_section,
        value_framework=strategy.value_framework,
        comparison_standard=strategy.comparison_standard,
        rhetorical_approach=strategy.rhetorical_approach,
        strategic_rationale=strategy.strategic_rationale,
        deep_evidence_section=deep_evidence_section,
        improvement_feedback=improvement_feedback,
    )

    llm = get_llm(provider="openai", temperature=0.8)  # Higher creativity for drafting

    logger.info("Calling LLM to draft opening statement...")
    response = await llm.ainvoke(prompt)

    # Extract draft text
    if hasattr(response, "content"):
        content = response.content
        # Handle both string and list content types
        if isinstance(content, list):
            draft_text = str(content)
        else:
            draft_text = content
    else:
        draft_text = str(response)

    draft_text = draft_text.strip()

    logger.info(f"Draft completed: {len(draft_text)} characters")

    # Save to filesystem
    filesystem.write("/draft.txt", draft_text)  # type: ignore[attr-defined]

    return {"draft": draft_text}
