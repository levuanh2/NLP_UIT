"""Evaluation data models."""

from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    question_id: str
    question: str
    expected_answer: str | None = None
    relevant_child_ids: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    sample_count: int = 0
