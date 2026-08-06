"""Answer generation models."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_name: str | None
    document_number: str | None
    article: str | None
    clause: str | None
    point: str | None
    evidence_id: str | None


class GeneratedAnswer(BaseModel):
    question_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    grounded: bool | None
    confidence: float | None
    abstained: bool = False
