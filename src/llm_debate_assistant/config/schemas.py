from pydantic import BaseModel, Field
from typing import List, Literal


class KeywordDefinition(BaseModel):
    keyword: str
    definition: str


class ArgumentsPrep(BaseModel):
    argument: str
    warrant: str
    evidence_needed: list[str]


class OpeningStatementOutline(BaseModel):
    keywords_definitions: list[KeywordDefinition]
    weighing_criterion: str
    arguments: list[ArgumentsPrep]


class ExampleCard(BaseModel):
    title: str
    url: str
    keypoints: list[str]
    original_text: list[str]


class Examples(BaseModel):
    argument: str
    warrant: str
    evidences: list[ExampleCard]


class OpeningStatement(BaseModel):
    opening_statement: str
    evidences_used: list[ExampleCard]


class OpeningStatementEvaluationFeedback(BaseModel):
    feedback: str
    evaluation_result: List[Literal["pass", "fail"]]


class RebuttalBase(BaseModel):
    opponent_statement: str
    rebuttal: str


class Rebuttal(BaseModel):
    rebuttals: list[RebuttalBase]


class JudgeComment(BaseModel):
    pro_score: int = Field(ge=0, le=100)
    con_score: int = Field(ge=0, le=100)
    feedback: str
