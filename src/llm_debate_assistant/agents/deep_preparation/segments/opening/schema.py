# -*- coding: utf-8 -*-
"""
Schema for opening statement segment.

This version leverages topic_research output for strategic selection and composition.
"""

from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    TopicResearchResult,
)


class SelectedKeyTerm(BaseModel):
    """A key term selected for use in opening statement."""

    term: str = Field(description="The key term")
    our_definition: str = Field(description="Our strategic definition to use")
    rationale: str = Field(description="Why this term is important for opening")


class SelectedArgument(BaseModel):
    """An argument selected from research for opening statement."""

    claim: str = Field(description="The argument claim")
    reasoning: str = Field(description="Logical reasoning supporting the claim")
    logical_chain: str = Field(description="Step-by-step logic chain")
    selection_rationale: str = Field(
        description="Why this argument was selected (strength, strategic fit, etc.)"
    )
    order: int = Field(description="Order to present (1, 2, or 3)")


class OpeningStrategy(BaseModel):
    """Strategic plan for opening statement."""

    selected_key_terms: list[SelectedKeyTerm] = Field(
        description="2-3 key terms to define",
        min_length=2,
        max_length=3,
    )
    selected_arguments: list[SelectedArgument] = Field(
        description="Exactly 3 arguments to present",
        min_length=3,
        max_length=3,
    )
    value_framework: str = Field(description="Value framework to emphasize")
    comparison_standard: str = Field(description="How to frame success/evaluation")
    rhetorical_approach: str = Field(
        description="Overall rhetorical strategy (e.g., assertive, preemptive, defensive)"
    )
    strategic_rationale: str = Field(
        description="Why this combination of terms + arguments works well together"
    )


class EvidenceSource(BaseModel):
    """Single evidence source with details."""

    url: str = Field(description="Source URL")
    title: str = Field(description="Source title")
    snippet: str = Field(description="Relevant snippet from source")
    credibility_note: str = Field(description="Why this source is credible")


class ArgumentEvidence(BaseModel):
    """Deep evidence for one selected argument."""

    argument_claim: str = Field(description="The argument this evidence supports")
    sources: list[EvidenceSource] = Field(
        description="8-12 high-quality sources",
        min_length=8,
        max_length=12,
    )
    best_quotes: list[str] = Field(
        description="3-5 most impactful quotes to use",
        max_length=5,
    )
    statistics: list[str] = Field(
        description="Key data points and statistics",
        default_factory=list,
    )
    case_studies: list[str] = Field(
        description="Real-world examples and case studies",
        default_factory=list,
    )


class EvaluationResult(BaseModel):
    """Evaluation of drafted opening statement."""

    result: Literal["pass", "fail"] = Field(description="Pass or fail")
    score: int = Field(description="Quality score 1-10", ge=1, le=10)
    strengths: list[str] = Field(description="What the statement does well")
    weaknesses: list[str] = Field(description="Areas needing improvement")
    feedback: str = Field(
        description="Detailed feedback for improvement (if fail)",
        default="",
    )
    strategy_alignment: bool = Field(description="Whether statement aligns with selected strategy")


class OpeningState(TypedDict):
    """State for opening statement workflow."""

    # Input
    topic: str
    side: Literal["正方", "反方"]
    filesystem: object  # Filesystem instance

    # Phase outputs
    research_context: Optional[TopicResearchResult]
    opening_strategy: Optional[OpeningStrategy]
    deep_evidence: Optional[list[ArgumentEvidence]]
    draft: Optional[str]
    evaluation: Optional[dict]

    # Control
    iteration_count: int
    max_iterations: int


class OpeningResult(BaseModel):
    """Final result from opening segment."""

    topic: str
    side: Literal["正方", "反方"]
    strategy: OpeningStrategy
    draft: str
    evaluation: EvaluationResult
    iterations: int

    def to_context_dict(self) -> dict:
        """Convert to dict for storage."""
        return self.model_dump()
