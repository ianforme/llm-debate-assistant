# -*- coding: utf-8 -*-
"""
Pydantic schemas for topic research agent.

This module defines the structured outputs for all research operations.
"""

from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


class KeyTerm(BaseModel):
    """Strategic definition of a key term in the debate topic."""

    term: str = Field(description="The key term to define")
    neutral_definition: str = Field(description="Objective, neutral definition")
    our_side_definition: str = Field(description="Definition favoring our side")
    opponent_definition: str = Field(description="Definition opponent might use")
    strategic_note: str = Field(description="Why our definition is better and how to use it")


class Argument(BaseModel):
    """Single argument with reasoning and evidence."""

    claim: str = Field(description="Main claim/assertion")
    reasoning: str = Field(description="Logical reasoning supporting the claim (200-300 words)")
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence (facts, data, examples)",
    )
    logical_chain: str = Field(description="Step-by-step logic (A → B → C → Conclusion)")
    strength: Literal["strong", "medium", "weak"] = Field(
        default="medium", description="Assessed strength of this argument"
    )


class PerspectiveResearch(BaseModel):
    """Research for one side of the debate."""

    side: Literal["正方", "反方"] = Field(description="Which side this research is for")
    core_claims: List[str] = Field(
        description="3-5 core claims that form the foundation of this side"
    )
    arguments: List[Argument] = Field(description="Detailed arguments with evidence")
    value_framework: str = Field(description="Underlying value system (e.g., freedom vs security)")
    comparison_standard: str = Field(
        description="How this side measures success/evaluates the debate"
    )


class KeyClash(BaseModel):
    """A point where the two sides' arguments directly conflict."""

    issue: str = Field(description="What the conflict is about")
    our_position: str = Field(description="Our side's position")
    opponent_position: str = Field(description="Opponent's position")
    analysis: str = Field(description="Analysis of who has the stronger argument and why")


class ComparativeAnalysis(BaseModel):
    """Strategic analysis comparing both sides."""

    key_clashes: List[KeyClash] = Field(description="3-5 points where arguments directly conflict")
    our_advantages: List[str] = Field(description="3-5 strengths in our argumentation")
    opponent_vulnerabilities: List[str] = Field(
        description="3-5 weaknesses in opponent's argumentation"
    )
    strategic_recommendations: List[str] = Field(
        description="5-7 specific recommendations for opening statement"
    )


class KeyTermsList(BaseModel):
    """Wrapper for list of key terms (for structured LLM output)."""

    terms: List[KeyTerm] = Field(description="List of key terms")


class CoreClaimsList(BaseModel):
    """Wrapper for list of core claims (for structured LLM output)."""

    claims: List[str] = Field(description="List of core claims")


class EvidenceList(BaseModel):
    """Wrapper for list of evidence strings (for structured LLM output)."""

    evidence: List[str] = Field(description="List of evidence")


class ValueFramework(BaseModel):
    """Wrapper for value framework and comparison standard (for structured LLM output)."""

    value_framework: str = Field(description="Underlying value system")
    comparison_standard: str = Field(description="How this side measures success")


class TopicResearchResult(BaseModel):
    """Complete output from topic research agent."""

    topic: str = Field(description="The debate topic")
    our_side: Literal["正方", "反方"] = Field(description="Which side we are arguing for")
    key_terms: List[KeyTerm] = Field(description="Strategic definitions of key terms")
    our_research: PerspectiveResearch = Field(description="Research for our side")
    opponent_research: PerspectiveResearch = Field(description="Research for opponent's side")
    analysis: ComparativeAnalysis = Field(description="Comparative strategic analysis")

    def to_context_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage in state context."""
        return self.model_dump()
