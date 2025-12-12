# -*- coding: utf-8 -*-
"""
Draft constructive speech operation.

Creates the constructive speech based on selected strategy and deep evidence.
Implements a "Writer -> Refiner" workflow to ensure high quality content
within strict length constraints.
"""

import logging
from typing import List, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.prompt_manager import get_prompt_manager
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech.schema import (
    ArgumentEvidence,
    ConstructiveStrategy,
    CritiqueResult,
)
from llm_debate_assistant.agents.deep_preparation.shared.utils import count_visible_chars

logger = logging.getLogger(__name__)


async def draft_constructive_speech(
    topic: str,
    side: Literal["正方", "反方"],
    strategy: ConstructiveStrategy,
    deep_evidence: List[ArgumentEvidence],
    iteration: int = 0,
    critique: Optional[CritiqueResult] = None,
) -> str:
    """Draft the constructive speech based on strategy and deep evidence.

    Orchestrates a Writer -> Refiner workflow:
    1. Writer (GPT): Generates rich content based on strategy and evidence.
    (Observed that GPT models have trouble counting Chinese characters accurately)

    2. Refiner (Gemini): Prunes the text if it exceeds word limits,
       ensuring specific evidence and logic are preserved.

    Args:
        topic (str): The debate topic.
        side (Literal["正方", "反方"]): Which side we are arguing for.
        strategy (ConstructiveStrategy): The constructive strategy.
        deep_evidence (List[ArgumentEvidence]): Evidence for each argument.
        iteration (int): Current iteration count (for iterative refinement).
        critique (Optional[CritiqueResult]): Optional critique from previous iteration.

    Returns:
        str: The drafted constructive speech.
    """
    logger.info(
        f"✍️ Drafting Constructive Speech (Iteration {iteration + 1}) for {topic} ({side})..."
    )

    # ------------------------------------------------------------------
    # 2. Format Context for Prompt
    # ------------------------------------------------------------------

    # A. Key Terms (Framing)
    key_terms_section = "\n".join(
        f"- **{t.term}**\n  定义: {t.definition_used}\n  战略用途: {t.strategic_usage}"
        for t in strategy.selected_key_terms
    )

    # B. Arguments & Evidence Mapping
    arguments_section = ""
    evidence_section = ""

    sorted_args = sorted(strategy.selected_arguments, key=lambda x: x.order)

    for i, arg_strat in enumerate(sorted_args):
        # Ensure we don't go out of bounds if deep_evidence is missing items
        arg_ev: ArgumentEvidence = deep_evidence[i] if i < len(deep_evidence) else None

        arguments_section += f"""
        **论点 {arg_strat.order}: {arg_strat.claim}**
        - 战术角色: {arg_strat.role} (请严格执行此角色的写作要求)
        - 逻辑推导 (Warrant): {arg_strat.warrant}
        - 影响/价值 (Impact): {arg_strat.impact}
        """

        if arg_ev:
            # Provide cleaned evidence
            stats_text = (
                "\n".join([f"    - {s}" for s in arg_ev.statistics[:3]])
                if arg_ev.statistics
                else "    (无具体数据)"
            )
            cases_text = (
                "\n".join([f"    - {c}" for c in arg_ev.case_studies[:2]])
                if arg_ev.case_studies
                else "    (无具体案例)"
            )
            quotes_text = (
                "\n".join([f"    - {q}" for q in arg_ev.best_quotes[:2]])
                if arg_ev.best_quotes
                else "    (无名言引用)"
            )

            evidence_section += f"""
            **[Source ID Group {i+1}] 用于支持论点 {arg_strat.order}**
            【数据弹药】:
            {stats_text}
            【实证案例】:
            {cases_text}
            【权威语录】:
            {quotes_text}
            ------------------------------------------------
            """
        else:
            evidence_section += (
                f"**论点 {arg_strat.order}**: 未找到强力证据，请侧重纯逻辑推演。\n"
            )

    # C. Feedback Integration (Iterative Refinement)
    improvement_instruction = ""

    if critique and iteration > 0:
        logger.info(f"Applying critique feedback (Score: {critique.score})")
        improvement_instruction = f"""
        【⚠️ 修改指令 (第 {iteration} 次修正)】
        上一版草稿评分: {critique.score}/10
        严重问题: {critique.critical_issues}
        具体建议: {critique.suggestions}

        请务必针对上述问题进行修正！
        """

    # ------------------------------------------------------------------
    # 3. Phase 1: Writer (Creative Generation)
    # ------------------------------------------------------------------

    pm = get_prompt_manager()
    prompt = pm.get("WRITING_CONSTRUCTIVE_SPEECH_PROMPT").format(
        topic=topic,
        side=side,
        speech_tone=strategy.speech_tone,
        key_terms_section=key_terms_section,
        comparison_standard=strategy.comparison_standard,
        arguments_section=arguments_section,
        evidence_section=evidence_section,
        value_premise=strategy.value_premise,
        strategic_alignment=strategy.strategic_alignment,
        improvement_instruction=improvement_instruction,
    )

    # Use GPT-4o for high-quality logic and rhetoric
    llm_writer = get_llm(provider="openai", temperature=0.7)

    logger.info("Step 1: Generating initial speech draft...")
    response = await llm_writer.ainvoke(prompt)

    draft_text = response.content if hasattr(response, "content") else str(response)
    draft_text = draft_text.strip()

    # ------------------------------------------------------------------
    # 4. Phase 2: Refiner (Constraint Enforcement)
    # ------------------------------------------------------------------

    current_chars = count_visible_chars(draft_text)
    MAX_ALLOWED_CHARS = 1300  # Threshold to trigger pruning
    TARGET_CHARS_RANGE = "1100-1150"

    if current_chars > MAX_ALLOWED_CHARS:
        logger.warning(
            f"⚠️ Draft too long ({current_chars} chars). Activating Refiner..."
        )

        # Use Gemini Flash for fast, cost-effective rewriting
        # Note: You can also use "openai" if preferred, but Flash is great for this.
        llm_refiner = get_llm(provider="gemini", temperature=0.1)

        # TODO: Move to managed prompts
        refine_prompt = ChatPromptTemplate.from_template(
            """
        # Role
        你是一位专业的文字编辑。

        # Task
        将以下辩论稿进行"无损压缩"。
        当前字数：{current_chars}
        目标字数：{TARGET_CHARS_RANGE}

        # Guidelines (严格遵守)
        1. **核心不改**：Hook/Pivot/Anchor 的结构、逻辑流向严禁修改。
        2. **数据保留**：文中引用的所有统计数据、年份、来源严禁删除。
        3. **删减冗余**：
           - 删除过度的形容词和铺垫。
           - 将啰嗦的长难句改写为精炼的短句。
           - 合并重复的语义。

        # Original Draft
        {draft_text}
        
        # Output
        直接输出修改后的正文，不要包含任何前言或说明。
        """
        )

        # refined_response = await llm_refiner.ainvoke(refine_prompt)
        # refined_text = (
        #     refined_response.content
        #     if hasattr(refined_response, "content")
        #     else str(refined_response)
        # )
        # refined_text = refined_text.strip()

        refiner_chain = refine_prompt | llm_refiner | StrOutputParser()

        refined_text = await refiner_chain.ainvoke(
            {
                "current_chars": current_chars,
                "TARGET_CHARS_RANGE": TARGET_CHARS_RANGE,
                "draft_text": draft_text,
            }
        )

        logger.info(
            f"✂️ Pruning complete. New length: {count_visible_chars(refined_text)}"
        )
        draft_text = refined_text
    else:
        logger.info(f"✅ Length within limits ({current_chars} chars).")

    # ------------------------------------------------------------------
    # 5. Return Final Draft
    # ------------------------------------------------------------------

    return draft_text
