# -*- coding: utf-8 -*-
"""
Pydantic schemas for topic research agent.

This module defines the structured outputs for all research operations.
"""

from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


class StrategicDefinition(BaseModel):
    """Strategic framing of a key term."""

    definition: str = Field(description="Strategic interpretation favoring our side")
    authority_anchor: str = Field(
        description="Citation, authority, or logical basis supporting this definition"
    )
    inclusion_exclusion: str = Field(
        description="What is included and deliberately excluded by this definition"
    )


class KeyTerm(BaseModel):
    """Strategic definition of a key term in the debate topic."""

    term: str = Field(description="The key concept to define")
    standard_definition: str = Field(
        description="Academic or commonly accepted definition"
    )
    strategic_definition: StrategicDefinition = Field(
        description="Our strategic framing with authority and scope"
    )
    opponents_trap: str = Field(
        description="Predicted opponent's definition trap and its harm"
    )
    burden_shift: str = Field(
        description="How burden of proof shifts with our definition"
    )


class ArgumentDraft(BaseModel):
    """Output for lightweight argument validation."""

    type: Literal["Value", "Practical"] = Field(
        description="Argument type: Value (principle-based) or Practical (utility-based)"
    )
    reasoning: str = Field(
        description="A logical sketch of the argument. Mark unverified assumptions with brackets."
    )
    logical_chain: str = Field(description="Simple causal chain (A -> B -> C).")
    impact: str = Field(
        description="Ultimate benefit or problem solved - whose interests are protected"
    )
    search_queries: List[str] = Field(
        description="1-2 broad keywords just to check if this topic exists/is debated."
    )


class Argument(BaseModel):
    """Single argument with type, claim, warrant, impact, and evidence."""

    type: Literal["Value", "Practical"] = Field(
        description="Argument type: Value (principle-based) or Practical (utility-based)"
    )
    claim: str = Field(description="Core assertion - short, punchy, memorable")
    warrant: str = Field(
        description="Logical reasoning explaining why the claim holds (mechanism/warrant)"
    )
    impact: str = Field(
        description="Ultimate benefit or problem solved - whose interests are protected"
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="List of evidence entries from web search, formatted with sources and content",
    )


class PerspectiveResearch(BaseModel):
    """Research for one side of the debate."""

    side: Literal["正方", "反方"] = Field(description="Which side this research is for")
    core_claims: List[str] = Field(
        description="3-5 core claims that form the foundation of this side"
    )
    arguments: List[Argument] = Field(description="Detailed arguments with evidence")
    value_framework: str = Field(
        description="Underlying value system (e.g., freedom vs security)"
    )
    comparison_standard: str = Field(
        description="How this side measures success/evaluates the debate"
    )


class KeyClash(BaseModel):
    """Represents a specific point of conflict."""

    issue: str = Field(description="The core topic of the clash")
    clash_type: Literal["Fact", "Value", "Definition"] = Field(
        description="The nature of the dispute"
    )
    our_position: str = Field(description="Our side's position")
    opponent_position: str = Field(description="Opponent's position")
    resolution_strategy: str = Field(
        description="Detailed strategy to win this point (e.g., using specific evidence or value weighing)."
    )


class StrategicPoint(BaseModel):
    """An advantage or vulnerability."""

    point: str = Field(description="The core point (1 sentence)")
    explanation: str = Field(
        description="Why this is a strength/weakness and how to use/exploit it"
    )


class StrategicRecommendation(BaseModel):
    """Actionable instruction for the case writer."""

    category: Literal["Framing", "Ordering", "Pre-emption", "Impact", "Burden"] = Field(
        description="The type of strategic instruction"
    )
    instruction: str = Field(
        description="Specific instruction on what to write or how to structure the case."
    )


class ComparativeAnalysis(BaseModel):
    """Full strategic analysis output."""

    key_clashes: List[KeyClash] = Field(
        description="3-4 points where arguments directly conflict"
    )
    our_advantages: List[StrategicPoint] = Field(
        description="3 strongest cards we hold (leverage points)"
    )
    opponent_vulnerabilities: List[StrategicPoint] = Field(
        description="Structural weaknesses in opponent's case (Achilles' heel)"
    )
    strategic_recommendations: List[StrategicRecommendation] = Field(
        description="5-7 specific actionable instructions for case construction"
    )


class KeyTermsList(BaseModel):
    """Wrapper for list of key terms (for structured LLM output)."""

    terms: List[KeyTerm] = Field(description="List of key terms")


class CoreArgument(BaseModel):
    """A structured core argument with full logical structure."""

    type: Literal["Value", "Practical"] = Field(
        description="Type of the argument: Principle/Value based or Consequence/Practicality based."
    )
    claim: str = Field(description="The core claim statement (one concise sentence).")
    warrant: str = Field(
        description="The logical reasoning behind the claim (The 'Why')."
    )
    impact: str = Field(
        description="The ultimate significance or benefit (The 'So What')."
    )


class CoreClaimsOutput(BaseModel):
    """Output container for core claims with full argument structure."""

    claims: List[CoreArgument] = Field(
        description="List of 3-5 distinct core arguments."
    )


class CoreClaimsList(BaseModel):
    """Wrapper for list of core claims (for structured LLM output).

    Deprecated: Use CoreClaimsOutput instead for richer structure.
    """

    claims: List[str] = Field(description="List of core claims")


class EvidenceList(BaseModel):
    """Wrapper for list of evidence strings (for structured LLM output)."""

    evidence: List[str] = Field(description="List of evidence")


class ValueFramework(BaseModel):
    """The philosophical framework for the case."""

    value_framework: str = Field(
        description="The ultimate moral goal or philosophical premise. Must be specific, catchy, and synthesized from the core claims. (30-60 words)"
    )

    comparison_standard: str = Field(
        description="The mechanism to weigh impacts. Explains which value takes precedence when values conflict (e.g., Rights > Economy). (30-60 words)"
    )


class TopicResearchResult(BaseModel):
    """Complete output from topic research agent."""

    topic: str = Field(description="The debate topic")
    our_side: Literal["正方", "反方"] = Field(
        description="Which side we are arguing for"
    )
    key_terms: List[KeyTerm] = Field(description="Strategic definitions of key terms")
    our_research: PerspectiveResearch = Field(description="Research for our side")
    opponent_research: PerspectiveResearch = Field(
        description="Research for opponent's side"
    )
    analysis: ComparativeAnalysis = Field(description="Comparative strategic analysis")

    def to_context_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage in state context."""
        return self.model_dump()
