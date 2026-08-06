"""Strict submission output models."""

from pydantic import BaseModel, ConfigDict


class SubmissionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


class SubmissionValidationResult(BaseModel):
    valid: bool
    errors: list[str]
    missing_question_ids: list[str]
    unexpected_question_ids: list[str]
