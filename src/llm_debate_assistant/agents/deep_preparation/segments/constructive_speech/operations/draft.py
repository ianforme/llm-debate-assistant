# -*- coding: utf-8 -*-
"""
Draft opening statement operation.

Creates the opening statement based on selected strategy and deep evidence.
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
)

logger = logging.getLogger(__name__)


async def draft_statement_node(state: OpeningState) -> dict[str, Any]:
    """Draft opening statement based on strategy and evidence.

    Creates a 4-minute opening statement (max 1200 visible characters):
    - Visible characters = Chinese chars + punctuation + English (no spaces/newlines)
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
        类型：{arg.type} ({"价值论证" if arg.type == "Value" else "实利论证"})
        推导逻辑：{arg.warrant}
        影响/实利：{arg.impact}
        选择理由：{arg.selection_rationale}
        """
        for arg in sorted(strategy.selected_arguments, key=lambda x: x.order)
    )

    # Deep evidence section - provide raw search results for model to extract from
    deep_evidence_section = chr(10).join(
        f"""
        **论点{i+1}证据：{ev.argument_claim}**

        来源数量：{len(ev.sources)}个搜索结果

        搜索结果原文（请从中提取相关引用、统计数据、案例）：
        {chr(10).join(ev.best_quotes) if ev.best_quotes else "（无搜索结果）"}

        来源列表：
        {chr(10).join(f"{j+1}. {src.title}" for j, src in enumerate(ev.sources[:8]))}
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

    # Use Gemini for better Chinese character counting and constraint adherence
    llm = get_llm(provider="gemini", temperature=0.8)

    logger.info("Calling Gemini LLM to draft opening statement...")
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

    char_count = count_visible_chars(draft_text)
    logger.info(
        f"Draft completed: {char_count} visible characters, {len(draft_text)} total"
    )

    # Save to filesystem (in /opening/ subfolder for organization)
    filesystem.write("/constructive_speech/draft.txt", draft_text)  # type: ignore[attr-defined]

    return {"draft": draft_text}
