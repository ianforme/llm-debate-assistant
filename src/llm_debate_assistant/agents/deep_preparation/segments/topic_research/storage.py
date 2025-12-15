# -*- coding: utf-8 -*-
"""
Storage functions for topic research.

Handles saving and loading research results to/from filesystem.

They now work deterministically for saving and loading research data.
"""

import json
from typing import Any, Literal, Optional

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    TopicResearchResult,
)
from llm_debate_assistant.agents.deep_preparation.schema import Filesystem
from llm_debate_assistant.agents.deep_preparation.storage import save_session_metadata


def save_research_to_filesystem(research: TopicResearchResult, filesystem: Filesystem) -> None:
    """Save topic research results to filesystem.

    Saves both structured JSON and human-readable markdown formats.

    Args:
        research (TopicResearchResult): Complete research result to save
        filesystem (Filesystem): Filesystem instance to write to

    File structure created:
        /research/
            research.json           # Full structured data
            research_summary.md     # Overview with key insights
            key_terms.md           # Strategic term definitions
            our_arguments.md       # Our side's arguments
            opponent_arguments.md  # Opponent's arguments
            analysis.md           # Comparative analysis
    """
    # Save session metadata for cache lookup
    save_session_metadata(
        filesystem=filesystem,
        topic=research.topic,
        side=research.our_side,
        segment="topic_research",
    )

    # Save full structured JSON for programmatic access
    filesystem.write(
        "/research/research.json",
        json.dumps(research.model_dump(), ensure_ascii=False, indent=2),
    )

    # Also save to final location (for backward compatibility)
    _save_research_files(research, filesystem)

    # Save summary markdown
    summary_parts = ["# Topic Research Summary\n"]
    summary_parts.append(f"**Topic**: {research.topic}")
    summary_parts.append(f"**Our Side**: {research.our_side}\n")
    summary_parts.append(f"- **Key Terms**: {len(research.key_terms)}")
    summary_parts.append(f"- **Our Arguments**: {len(research.our_research.arguments)}")
    summary_parts.append(f"- **Opponent Arguments**: {len(research.opponent_research.arguments)}")
    summary_parts.append(f"- **Key Clashes**: {len(research.analysis.key_clashes)}")
    summary_parts.append(
        f"- **Strategic Recommendations**: {len(research.analysis.strategic_recommendations)}\n"
    )

    filesystem.write("/research/research_summary.md", "\n".join(summary_parts))

    # Save key terms
    terms_parts = ["# Key Terms (Strategic Definitions)\n"]
    for term in research.key_terms:
        terms_parts.append(f"## {term.term}\n")
        terms_parts.append(f"**基准定义**: {term.standard_definition}\n")
        terms_parts.append(f"**我方战略定义**: {term.strategic_definition.definition}\n")
        terms_parts.append(f"**权威锚点**: {term.strategic_definition.authority_anchor}\n")
        terms_parts.append(f"**收纳与切割**: {term.strategic_definition.inclusion_exclusion}\n")
        terms_parts.append(f"**对方定义的陷阱**: {term.opponents_trap}\n")
        terms_parts.append(f"**举证责任转移**: {term.burden_shift}\n")

    filesystem.write("/research/key_terms.md", "\n".join(terms_parts))

    # Save our arguments
    our_parts = [f"# Our Side ({research.our_side}) Research\n"]
    our_parts.append("## Core Claims\n")
    for i, claim in enumerate(research.our_research.core_claims, 1):
        our_parts.append(f"{i}. {claim}")

    our_parts.append(f"\n## Value Framework\n{research.our_research.value_framework}\n")
    our_parts.append(f"## Comparison Standard\n{research.our_research.comparison_standard}\n")

    our_parts.append("## Arguments\n")
    for i, arg in enumerate(research.our_research.arguments, 1):
        our_parts.append(f"### Argument {i}: {arg.claim}\n")
        our_parts.append(f"**Type**: {arg.type} (价值论证 or 实利论证)\n")
        our_parts.append(f"**Warrant/Mechanism**: {arg.warrant}\n")
        our_parts.append(f"**Impact**: {arg.impact}\n")
        our_parts.append("")

    filesystem.write("/research/our_arguments.md", "\n".join(our_parts))

    # Save opponent arguments
    opponent_side = "反方" if research.our_side == "正方" else "正方"
    opp_parts = [f"# Opponent ({opponent_side}) Research\n"]
    opp_parts.append("## Core Claims\n")
    for i, claim in enumerate(research.opponent_research.core_claims, 1):
        opp_parts.append(f"{i}. {claim}")

    opp_parts.append(f"\n## Value Framework\n{research.opponent_research.value_framework}\n")
    opp_parts.append(f"## Comparison Standard\n{research.opponent_research.comparison_standard}\n")

    opp_parts.append("## Arguments\n")
    for i, arg in enumerate(research.opponent_research.arguments, 1):
        opp_parts.append(f"### Argument {i}: {arg.claim}\n")
        opp_parts.append(f"**Type**: {arg.type} (价值论证 or 实利论证)\n")
        opp_parts.append(f"**Warrant/Mechanism**: {arg.warrant}\n")
        opp_parts.append(f"**Impact**: {arg.impact}\n")
        opp_parts.append("")

    filesystem.write("/research/opponent_arguments.md", "\n".join(opp_parts))

    # Save comparative analysis
    analysis_parts = ["# Comparative Analysis\n"]

    analysis_parts.append("## Key Clashes\n")
    for i, clash in enumerate(research.analysis.key_clashes, 1):
        analysis_parts.append(f"### Clash {i} ({clash.clash_type}): {clash.issue}\n")
        analysis_parts.append(f"**Our Position**: {clash.our_position}\n")
        analysis_parts.append(f"**Opponent Position**: {clash.opponent_position}\n")
        analysis_parts.append(f"**Resolution Strategy**: {clash.resolution_strategy}\n\n")

    analysis_parts.append("## Our Advantages\n")
    for i, adv in enumerate(research.analysis.our_advantages, 1):
        analysis_parts.append(f"{i}. **{adv.point}**\n")
        analysis_parts.append(f"   {adv.explanation}\n\n")

    analysis_parts.append("\n## Opponent Vulnerabilities\n")
    for i, vuln in enumerate(research.analysis.opponent_vulnerabilities, 1):
        analysis_parts.append(f"{i}. **{vuln.point}**\n")
        analysis_parts.append(f"   {vuln.explanation}\n\n")

    analysis_parts.append("\n## Strategic Recommendations\n")
    for i, rec in enumerate(research.analysis.strategic_recommendations, 1):
        analysis_parts.append(f"{i}. [{rec.category}] {rec.instruction}\n\n")

    filesystem.write("/research/analysis.md", "\n".join(analysis_parts))


def _save_research_files(research: TopicResearchResult, filesystem: Filesystem) -> None:
    """Helper to save markdown files from research result."""
    # (Extracted to avoid duplication - called by both full and partial saves)
    pass  # Files already saved above, this is for future refactoring


# ============================================================================
# Incremental Save/Load Functions (for cache/resume)
# ============================================================================


def save_key_terms_progress(
    topic: str,
    side: Literal["正方", "反方"],
    key_terms: list,
    filesystem: Filesystem,
) -> None:
    """Save key terms progress for resumability.

    Args:
        topic (str): Debate topic
        side (str): Our side
        key_terms (list): List of KeyTerm objects
        filesystem (Filesystem): Filesystem instance
    """

    progress_data = {
        "topic": topic,
        "side": side,
        "key_terms": [
            term.model_dump() if hasattr(term, "model_dump") else term for term in key_terms
        ],
        "stage": "terms_complete",
    }

    filesystem.write(
        "/research/_progress.json",
        json.dumps(progress_data, ensure_ascii=False, indent=2),
    )


def save_research_progress(
    topic: str,
    side: Literal["正方", "反方"],
    key_terms: list,
    our_research: Any,
    opponent_research: Any,
    filesystem: Filesystem,
) -> None:
    """Save research progress for resumability.

    Args:
        topic (str): Debate topic
        side (str): Our side
        key_terms (list): List of KeyTerm objects
        our_research (Any): PerspectiveResearch for our side
        opponent_research (Any): PerspectiveResearch for opponent
        filesystem (Filesystem): Filesystem instance
    """
    progress_data = {
        "topic": topic,
        "side": side,
        "key_terms": [
            term.model_dump() if hasattr(term, "model_dump") else term for term in key_terms
        ],
        "our_research": (
            our_research.model_dump() if hasattr(our_research, "model_dump") else our_research
        ),
        "opponent_research": (
            opponent_research.model_dump()
            if hasattr(opponent_research, "model_dump")
            else opponent_research
        ),
        "stage": "research_complete",
    }

    filesystem.write(
        "/research/_progress.json",
        json.dumps(progress_data, ensure_ascii=False, indent=2),
    )


def save_analysis_progress(
    topic: str,
    side: Literal["正方", "反方"],
    key_terms: list,
    our_research: Any,
    opponent_research: Any,
    analysis: Any,
    filesystem: Filesystem,
) -> None:
    """Save analysis progress for resumability.

    Args:
        topic (str): Debate topic
        side (str): Our side
        key_terms (list): List of KeyTerm objects
        our_research (Any): PerspectiveResearch for our side
        opponent_research (Any): PerspectiveResearch for opponent
        analysis (Any): ComparativeAnalysis result
        filesystem (Filesystem): Filesystem instance
    """
    progress_data = {
        "topic": topic,
        "side": side,
        "key_terms": [
            term.model_dump() if hasattr(term, "model_dump") else term for term in key_terms
        ],
        "our_research": (
            our_research.model_dump() if hasattr(our_research, "model_dump") else our_research
        ),
        "opponent_research": (
            opponent_research.model_dump()
            if hasattr(opponent_research, "model_dump")
            else opponent_research
        ),
        "analysis": (analysis.model_dump() if hasattr(analysis, "model_dump") else analysis),
        "stage": "analysis_complete",
    }

    filesystem.write(
        "/research/_progress.json",
        json.dumps(progress_data, ensure_ascii=False, indent=2),
    )


def load_research_progress(filesystem: Filesystem) -> Optional[dict]:
    """Load partial research progress if it exists.
    Args:
        filesystem (Filesystem): Filesystem instance to read from
    Returns:
        Progress dict with 'stage' field indicating completion level, or None
    """
    try:
        result = filesystem.read("/research/_progress.json")
        if not result.get("success") or not result.get("content"):
            return None

        return json.loads(result["content"])

    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def load_research_from_filesystem(
    filesystem: Filesystem,
) -> Optional[TopicResearchResult]:
    """Load cached topic research from filesystem if it exists.

    Args:
        filesystem (Filesystem): Filesystem instance to read from

    Returns:
        TopicResearchResult: TopicResearchResult
        if cached research exists, None otherwise
    """
    try:
        result = filesystem.read("/research/research.json")
        if not result.get("success") or not result.get("content"):
            return None

        research_data = json.loads(result["content"])
        return TopicResearchResult(**research_data)

    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def get_research_cache_key(topic: str, our_side: Literal["正方", "反方"]) -> str:
    """Generate cache key for research results.

    Args:
        topic (str): Debate topic
        our_side (Literal["正方", "反方"]): Which side we're arguing for

    Returns:
        str: Cache key string
    """
    # Simple cache key - could be made more sophisticated with hashing
    return f"{topic}_{our_side}"
