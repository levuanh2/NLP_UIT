"""Retrieval and context models."""

from pydantic import BaseModel

from app.domain.metadata import LegalMetadata
from app.domain.queries import QueryMetadata


class RetrievalCandidate(BaseModel):
    child_id: str
    text: str
    metadata: LegalMetadata
    dense_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    query: str
    candidates: list[RetrievalCandidate]
    metadata_filter_applied: bool
    filter_metadata: QueryMetadata | None


class LegalEvidence(BaseModel):
    evidence_id: str
    document_id: int
    document_name: str | None
    source_link: str | None
    chapter: str | None
    section: str | None
    article: str | None
    clause: str | None
    point: str | None
    text: str


class LegalContext(BaseModel):
    query: str
    evidences: list[LegalEvidence]
    formatted_context: str
    token_count: int | None
