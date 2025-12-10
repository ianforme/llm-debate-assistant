# -*- coding: utf-8 -*-
"""
Utility functions for topic research operations.
"""

import json
from typing import List, Literal

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    Argument,
)


def opposite_side(side: Literal["正方", "反方"]) -> Literal["正方", "反方"]:
    """Get the side of your debate component

    Args:
        side (Literal["正方", "反方"]): The side of the debate ("正方" or "反方")

    Returns:
        Literal["正方", "反方"]: The opposite side of the debate ("正方" or "反方")
    """
    return "反方" if side == "正方" else "正方"


def format_arguments_truncated(arguments: List[Argument]) -> str:
    """Format arguments with truncated reasoning for use in prompts.

    Truncates reasoning to first 100 characters to keep prompt concise.

    Args:
        arguments (List[Argument]): List of Argument objects to format.

    Returns:
        str: A formatted string with truncated arguments.
    """
    formatted = []
    for i, arg in enumerate(arguments, 1):
        text = f"{i}. {arg.claim}\n   类型：{arg.type}\n   推导：{arg.warrant[:100]}..."
        text += f"\n   影响：{arg.impact[:80]}..."
        formatted.append(text)
    return "\n\n".join(formatted)


def extract_logical_chain(text: str) -> str:
    """Extract logical chain from reasoning text.

    Looks for arrow patterns (→, ->, -->) or numbered steps.
    Falls back to first sentence if no clear chain found.

    Args:
        text (str): The reasoning text to extract
        the logical chain from.

    Returns:
        str: The extracted logical chain or the
        first sentence as a fallback.
    """
    # Look for arrow patterns
    if "→" in text or "->" in text or "-->" in text:
        # Find lines with arrows
        lines = text.split("\n")
        chain_lines = [line.strip() for line in lines if "→" in line or "->" in line]
        if chain_lines:
            return " ".join(chain_lines)

    # Fallback: extract first sentence
    sentences = text.split("。")
    if sentences:
        return sentences[0] + "。"

    return text[:100]


def assess_strength(evidence: List[str]) -> Literal["strong", "medium", "weak"]:
    """Assess argument strength based on evidence count and quality.

    Args:
        evidence (List[str]): List of evidence pieces supporting the argument.
    Returns:
        Literal["strong", "medium", "weak"]: Strength assessment.

    Simple heuristic:
    - strong: 4+ pieces of evidence
    - medium: 2-3 pieces
    - weak: 0-1 pieces
    """
    evidence_count = len(evidence)

    if evidence_count >= 4:
        return "strong"
    elif evidence_count >= 2:
        return "medium"
    else:
        return "weak"


def extract_from_json_response(response_text: str) -> dict:
    """This is a fallback or error handling function
    to parse the output from LLM in case they are not
    well structured JSON.

    It attempts to extract JSON even if the LLM response
    includes additional text or formatting, which is not
    uncommon especially for Gemini.

    Args:
        response_text (str): LLM response

    Raises:
        ValueError: If JSON cannot be extracted from the response.

    Returns:
        dict: Parsed JSON object.
    """
    # Try direct parse first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Look for JSON in code blocks
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end > start:
            try:
                return json.loads(response_text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Look for JSON object pattern
    if "{" in response_text and "}" in response_text:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        try:
            return json.loads(response_text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {response_text[:200]}")
