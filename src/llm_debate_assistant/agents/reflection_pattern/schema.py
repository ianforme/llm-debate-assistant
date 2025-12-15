"""Pydantic models for structured LLM outputs."""

from typing import Literal

from pydantic import BaseModel, Field


class KeywordDefinition(BaseModel):
    """Definition of a key term from the debate topic."""

    keyword: str = Field(description="关键词（必须是辩题原文）")
    definition: str = Field(description="清晰的定义")


class ComparisonStandard(BaseModel):
    """Comparison standard for the debate."""

    standard: str = Field(description="比较标准")
    justification: str = Field(description="为何选择此标准")


class Argument(BaseModel):
    """Single debate argument with claim, warrant, and evidence needed."""

    claim: str = Field(description="论点")
    warrant: str = Field(description="论证")
    evidence_needed: list[str] = Field(description="所需证据类型列表")


class DebateOutline(BaseModel):
    """Complete debate outline structure."""

    keyword_definitions: list[KeywordDefinition]
    comparison_standard: ComparisonStandard
    arguments: list[Argument] = Field(max_length=3, description="最多3个论点")


class OpeningStatement(BaseModel):
    """Opening statement with evidence tracking."""

    content: str = Field(description="开篇立论内容（最多1200字）")
    evidences_used: list[str] = Field(description="使用的证据来源")
    word_count: int = Field(description="字数统计")


class Evaluation(BaseModel):
    """Evaluation result with feedback."""

    evaluation_result: Literal["pass", "fail"]
    feedback: str = Field(description="详细反馈")
    next_action: Literal["redo_outline", "redo_evidence", "redo_draft", "improve"] | None = Field(
        default=None, description="下一步行动（仅当evaluation_result为fail时需要）"
    )
