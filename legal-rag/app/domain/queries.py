"""Legal question and query-analysis models."""

from pydantic import BaseModel, Field


class LegalQuery(BaseModel):
    question_id: str
    question: str


class QueryMetadata(BaseModel):
    raw_query: str = ""
    document_name: str | None = None
    document_id: int | None = None
    document_number: str | None = None
    document_type: str | None = None
    issued_year: int | None = None
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    clause: str | None = None
    point: str | None = None
    legal_topic: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class QueryAnalysis(BaseModel):
    original_query: str
    normalized_query: str
    intent: str | None = None
    metadata: QueryMetadata
    should_apply_metadata_filter: bool
