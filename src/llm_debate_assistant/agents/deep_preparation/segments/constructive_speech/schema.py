# -*- coding: utf-8 -*-
"""
Schema for constructive speech segment.

This version leverages topic_research output for strategic selection and composition
of the constructive speech. It defines the data models for strategy, evidence,
drafts, critiques, and overall segment state.
"""

from typing import List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field, ConfigDict

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    TopicResearchResult,
)


class SelectedTerm(BaseModel):
    """A key term selected for the constructive speech."""

    term: str
    definition_used: str = Field(
        description="The specific definition text to be used in the speech."
    )
    strategic_usage: str = Field(
        description="How to use this definition to frame the debate (e.g., 'exclude X', 'shift burden')."
    )


class SelectedArgument(BaseModel):
    """An argument selected for the constructive speech."""

    claim: str
    warrant: str
    impact: str
    evidence_summary: str = Field(
        description="A brief summary of the key evidence to include."
    )
    order: int = Field(description="Position in the speech: 1, 2, or 3.")
    role: Literal["The Hook", "The Pivot", "The Anchor"] = Field(
        description="Strategic role: 'The Hook' (Strongest/Intuitive), 'The Pivot' (Pre-emptive/Strategic), 'The Anchor' (Value/Deep)."
    )
    rationale: str = Field(
        description="Why this argument was selected and placed in this order."
    )


class ConstructiveStrategy(BaseModel):
    """The strategic blueprint for the constructive speech."""

    selected_key_terms: List[SelectedTerm] = Field(
        description="2-3 terms to define for strategic framing."
    )

    selected_arguments: List[SelectedArgument] = Field(
        description="Exactly 3 arguments, ordered strategically."
    )

    value_premise: str = Field(
        description="The overarching moral/philosophical theme of the case."
    )
    comparison_standard: str = Field(
        description="The criterion for judging the round (e.g., 'Net Benefits', 'Rights Protection')."
    )

    speech_tone: str = Field(
        description="The rhetorical tone (e.g., 'Empathetic', 'Analytical', 'Urgent')."
    )

    strategic_alignment: str = Field(
        description="Explanation of how this strategy exploits opponent weaknesses and leverages our strengths."
    )


class SearchPlan(BaseModel):
    """Output schema for the Planning phase."""

    queries: List[str] = Field(
        description="3-5 highly specific search queries targeting dates, numbers, reports, and cases."
    )


class ExtractedItem(BaseModel):
    """Single evidence item."""

    text: str = Field(description="The content of the evidence.")
    source_id: int = Field(description="The Source ID.")


class SynthesizedEvidence(BaseModel):
    """Pydantic V2 version Schema."""

    model_config = ConfigDict(populate_by_name=True)

    statistics: List[ExtractedItem] = Field(default_factory=list, alias="Statistics")
    quotes: List[ExtractedItem] = Field(default_factory=list, alias="Quotes")
    case_studies: List[ExtractedItem] = Field(default_factory=list, alias="CaseStudies")


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
        description="8-12 high-quality sources (ideally, but accepts whatever is available)",
        max_length=15,  # Allow a bit more flexibility
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


class CritiqueResult(BaseModel):
    """Evaluation of drafted constructive speech."""

    score: int = Field(description="Overall score 1-10", ge=1, le=10)
    decision: Literal["pass", "needs_revision"] = Field(
        description="Whether the draft passes or needs revision"
    )

    strategy_compliance: str = Field(
        description="Comment on Hook/Pivot/Anchor execution"
    )
    evidence_usage: str = Field(
        description="Comment on whether specific provided evidence was used"
    )

    critical_issues: List[str] = Field(description="List of major failures to fix")
    suggestions: List[str] = Field(description="Specific instructions for the rewrite")


class ConstructiveState(TypedDict, total=False):
    """State for the Constructive Speech (Case Construction) workflow.

    Note: Non-serializable dependencies (filesystem) are passed via config["configurable"]
    for proper checkpointing support.
    """

    topic: str
    side: Literal["正方", "反方"]

    time_limit: str
    word_count_limit: int

    research_context: Optional[TopicResearchResult]

    constructive_strategy: Optional[ConstructiveStrategy]

    deep_evidence: Optional[List[ArgumentEvidence]]

    draft_content: Optional[str]

    critique: Optional[CritiqueResult]

    iteration_count: int


class ConstructiveResult(BaseModel):
    """Final result from constructive speech segment."""

    topic: str
    side: Literal["正方", "反方"]
    strategy: ConstructiveStrategy
    draft: str
    critique: CritiqueResult
    iterations: int

    def to_context_dict(self) -> dict:
        """Convert to dict for storage."""
        return self.model_dump()
