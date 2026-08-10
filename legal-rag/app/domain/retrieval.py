"""Retrieval, ranking, and context models."""

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.metadata import LegalMetadata
from app.domain.queries import QueryMetadata


class RetrievalCandidate(BaseModel):
    child_id: str
    score: float = 0.0
    source: Literal["dense", "bm25", "rrf", "reranker"] = "rrf"
    rank: int = Field(default=0, ge=0)
    # Small optional compatibility fields. Retrieval itself resolves chunk data from
    # SQLite, and does not build a corpus-sized metadata structure in memory.
    text: str | None = None
    metadata: LegalMetadata | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    query: str
    query_metadata: QueryMetadata
    candidates: list[RetrievalCandidate]
    evidences: list["LegalEvidence"]
    active_index_version: str
    dense_count: int
    bm25_count: int
    fused_count: int
    reranked_count: int
    metadata_filter_applied: bool = False
    metadata_filter_fallback: bool = False


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
    parent_id: str | None = None
    child_id: str | None = None
    position: int | None = None
    rank: int | None = None


class LegalContext(BaseModel):
    query: str
    evidences: list[LegalEvidence]
    formatted_context: str
    token_count: int | None
