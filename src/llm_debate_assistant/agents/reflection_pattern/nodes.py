from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from llm_debate_assistant.prompts import opening_statement_prompts
from llm_debate_assistant.services.llm import get_llm
from llm_debate_assistant.services.web_search import search_multiple_arguments  # type: ignore[attr-defined]

from .schema import DebateOutline, Evaluation, OpeningStatement
from .state import DebateState


# LLM call Node
async def create_outline_node(state: DebateState, config: RunnableConfig) -> dict[str, Any]:
    """Create debate outline with definitions, standards, and arguments.

    Args:
        state (DebateState): Current debate state
        config (RunnableConfig): Runnable configuration

    Raises:
        ValueError: If the LLM returns None.
        RuntimeError: If the LLM call fails.

    Returns:
        dict[str, Any]: Updated state with outline.
    """
    prompt = opening_statement_prompts.debate_outline_prompt(
        topic=state["topic"], side=state["side"]
    )

    # Add feedback if redoing outline
    # if state.get("evaluation") and state["evaluation"].get("feedback"):
    #     feedback = state["evaluation"]["feedback"]
    #     prompt += f"\n\n【评审反馈】\n{feedback}\n\n请根据以上反馈重新设计大纲。"

    llm = get_llm(temperature=0.7)
    structured_llm = llm.with_structured_output(DebateOutline)

    try:
        outline = await structured_llm.ainvoke(prompt)

        if outline is None:
            raise ValueError("LLM returned None - API call may have failed")

        message = "✅ Created debate outline"
        if state.get("evaluation"):
            message += " (revised based on feedback)"

        return {
            "outline": outline.model_dump(),
            "messages": [SystemMessage(content=message)],
        }
    except Exception as e:
        raise RuntimeError(f"Failed to create outline: {e}\nTopic: {state['topic']}") from e


# Tool Node: Model has been pre-configured in the search function
async def search_evidence_node(state: DebateState, config: RunnableConfig) -> dict[str, Any]:
    """Search for evidence supporting debate arguments.

    Args:
        state (DebateState): Current debate state
        config (RunnableConfig): Runnable configuration

    Returns:
        dict[str, Any]: Updated state with evidence data
    """
    outline = state["outline"]
    if outline is None:
        raise ValueError("Outline is required for evidence search")

    arguments = outline["arguments"]

    # Prepare search arguments
    search_args = [(arg["claim"], arg["warrant"], arg["evidence_needed"]) for arg in arguments]

    # Check if there's feedback from evaluation (when redoing evidence)
    feedback = None
    evaluation = state.get("evaluation")
    if evaluation and evaluation.get("feedback"):
        feedback = evaluation["feedback"]

    # Use thread-based concurrent search (seems faster + more reliable?)
    evidence_results = await search_multiple_arguments(
        arguments=search_args,
        topic=state["topic"],
        side=state["side"],
        use_threaded=True,
        feedback=feedback,
    )

    # Convert to dict format and store directly in state
    evidence_dicts = [
        {
            "argument": res.argument,
            "warrant": res.warrant,
            "search_results": res.results.get("search_results", []),
            "analysis": res.results.get("text", ""),
            "error": res.error,
        }
        for res in evidence_results
    ]

    total_sources = sum(len(ev.get("search_results", [])) for ev in evidence_dicts)

    # Add message indicating if feedback was used
    message = f"✅ Gathered evidence ({total_sources} sources)"
    if feedback:
        message += " (refined based on feedback)"

    return {
        "evidence_data": evidence_dicts,
        "messages": [SystemMessage(content=message)],
    }


# Another LLM call Node
async def draft_statement_node(state: DebateState, config: RunnableConfig) -> dict[str, Any]:
    """Draft opening statement using evidence.

    Args:
        state (DebateState): Current debate state
        config (RunnableConfig): Runnable configuration

    Returns:
        dict[str, Any]: Updated state with draft
    """
    outline = state["outline"]
    if outline is None:
        raise ValueError("Outline is required for drafting statement")

    evidence_data = state.get("evidence_data") or []

    # Build context with evidence
    debate_outline_parts = ["## 关键词定义"]
    for kw in outline["keyword_definitions"]:
        debate_outline_parts.append(f"- {kw['keyword']}: {kw['definition']}")

    debate_outline_parts.append(f"\n## 比较标准\n{outline['comparison_standard']['standard']}")
    debate_outline_parts.append(f"理由: {outline['comparison_standard']['justification']}\n")

    debate_outline_parts.append("## 论点与证据\n")

    # Add evidence for each argument
    for i, arg in enumerate(outline["arguments"]):
        debate_outline_parts.append(f"\n### 论点 {i + 1}")
        debate_outline_parts.append(f"**论点**: {arg['claim']}")
        debate_outline_parts.append(f"**论证**: {arg['warrant']}\n")

        # Add analysis from evidence
        if i < len(evidence_data):
            evidence = evidence_data[i]
            if evidence.get("analysis"):
                debate_outline_parts.append("**证据**:")
                debate_outline_parts.append(f"\n{evidence['analysis']}\n")

    debate_outline = "\n".join(debate_outline_parts)

    # Generate draft
    prompt = opening_statement_prompts.opening_statement_prompt(
        debate_outline=debate_outline,
        topic=state["topic"],
        side=state["side"],
    )

    # Add feedback if redoing draft
    evaluation = state.get("evaluation")
    if evaluation and evaluation.get("feedback"):
        feedback = evaluation["feedback"]
        prompt += f"\n\n【评审反馈】\n{feedback}\n\n请根据以上反馈重新撰写立论稿。"

    # Higher temp for creativity
    llm = get_llm(temperature=0.8)
    structured_llm = llm.with_structured_output(OpeningStatement)
    draft = await structured_llm.ainvoke(prompt)

    message = f"✅ Drafted opening statement ({draft.word_count} words)"
    if state.get("evaluation"):
        message += " (revised based on feedback)"

    return {
        "draft": draft.content,
        "messages": [SystemMessage(content=message)],
    }


async def evaluate_statement_node(state: DebateState, config: RunnableConfig) -> dict[str, Any]:
    """Evaluate opening statement quality.

    Args:
        state (DebateState): Current debate state with draft
        config (RunnableConfig): Runnable configuration

    Returns:
        dict[str, Any]: Updated state with evaluation
    """
    outline = state["outline"]
    if outline is None:
        raise ValueError("Outline is required for evaluation")

    evidence_data = state.get("evidence_data") or []

    # Build compact outline for evaluation
    debate_outline_parts = ["## 关键词定义"]
    for kw in outline["keyword_definitions"]:
        debate_outline_parts.append(f"- {kw['keyword']}: {kw['definition']}")

    debate_outline_parts.append(f"\n## 比较标准\n{outline['comparison_standard']['standard']}\n")

    debate_outline_parts.append("## 论点框架")
    for i, arg in enumerate(outline["arguments"], 1):
        debate_outline_parts.append(f"{i}. {arg['claim']}")

    # Add evidence summary
    if evidence_data:
        debate_outline_parts.append("\n## 可用证据")
        for i, ev in enumerate(evidence_data, 1):
            if ev.get("analysis"):
                # Show first 300 chars of analysis
                # Truncate to lower API cost/Token usage/response time
                preview = ev["analysis"][:300]
                if len(ev["analysis"]) > 300:
                    preview += "..."
                debate_outline_parts.append(f"\n论点 {i}: {preview}")

    debate_outline = "\n".join(debate_outline_parts)

    # Create evaluation prompt
    prompt = opening_statement_prompts.opening_statement_evaluator_prompt(
        debate_outline=debate_outline,
        topic=state["topic"],
        side=state["side"],
    )
    prompt += f"\n\n【待评估的开篇立论】\n{state['draft']}"

    # Low temp for consistency
    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(Evaluation)

    # Add iteration number to evaluation
    iteration = state.get("iteration_count", 1)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"这是第 {iteration} 次评估。"),
    ]

    evaluation = await structured_llm.ainvoke(messages)

    # Set next_action from evaluation, or default to "auto" if not provided
    next_action = evaluation.next_action if evaluation.next_action else "auto"

    return {
        "evaluation": {
            **evaluation.model_dump(),
            "iteration_number": iteration,
        },
        "next_action": next_action,
        "iteration_count": iteration + 1,  # Increment after each evaluation
        "messages": [
            SystemMessage(
                content=f"📊 Evaluation (iteration {iteration}): {evaluation.evaluation_result}"
            )
        ],
    }


async def improve_statement_node(state: DebateState, config: RunnableConfig) -> dict[str, Any]:
    """Improve opening statement based on evaluation feedback.

    Args:
        state (DebateState): Current debate state with draft
        config (RunnableConfig): Runnable configuration

    Returns:
        dict[str, Any]: Updated state with evaluation
    """
    outline = state["outline"]
    if outline is None:
        raise ValueError("Outline is required for improvement")

    evaluation = state["evaluation"]
    if evaluation is None:
        raise ValueError("Evaluation is required for improvement")

    evidence_data = state.get("evidence_data") or []
    feedback = evaluation["feedback"]

    # Build context with all available evidence
    debate_outline_parts = ["## 关键词定义"]
    for kw in outline["keyword_definitions"]:
        debate_outline_parts.append(f"- {kw['keyword']}: {kw['definition']}")

    debate_outline_parts.append(f"\n## 比较标准\n{outline['comparison_standard']['standard']}\n")

    debate_outline_parts.append("## 论点与完整证据\n")

    # Add all evidence for improvement
    for i, arg in enumerate(outline["arguments"]):
        debate_outline_parts.append(f"\n### 论点 {i + 1}")
        debate_outline_parts.append(f"**论点**: {arg['claim']}")
        debate_outline_parts.append(f"**论证**: {arg['warrant']}\n")

        if i < len(evidence_data):
            evidence = evidence_data[i]
            if evidence.get("analysis"):
                debate_outline_parts.append("**证据**:")
                debate_outline_parts.append(f"\n{evidence['analysis']}\n")

    debate_outline = "\n".join(debate_outline_parts)

    # Create improvement prompt
    prompt = opening_statement_prompts.opening_statement_improver_prompt(
        debate_outline=debate_outline,
        topic=state["topic"],
        side=state["side"],
    )
    prompt += f"\n\n【当前立论稿】\n{state['draft']}"
    prompt += f"\n\n【评审反馈】\n{feedback}"

    # High temp to reflect a more robust and diverse views
    llm = get_llm(temperature=0.8)
    structured_llm = llm.with_structured_output(OpeningStatement)
    improved_draft = await structured_llm.ainvoke(prompt)

    return {
        "draft": improved_draft.content,
        "messages": [
            SystemMessage(content=f"🔄 Improved draft ({improved_draft.word_count} words)")
        ],
    }
