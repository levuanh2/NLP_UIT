"""Optional, confidence-gated pre-retrieval metadata filtering."""

from pydantic import BaseModel, Field

from app.domain.queries import QueryMetadata
from app.indexing.metadata_store.repository import LegalRepository


class RetrievalFilter(BaseModel):
    candidate_ids: set[str] | None = None
    applied: bool = False
    authoritative: bool = False
    matched_count: int = Field(default=0, ge=0)
    fields: tuple[str, ...] = ()


class MetadataFilter:
    def __init__(
        self,
        repository: LegalRepository,
        enabled: bool = True,
        min_confidence: float = 0.8,
        fallback_to_full_corpus: bool = True,
    ) -> None:
        self.repository = repository
        self.enabled = enabled
        self.min_confidence = min_confidence
        self.fallback_to_full_corpus = fallback_to_full_corpus

    def build_filter(self, metadata: QueryMetadata) -> RetrievalFilter:
        """Convert only explicit metadata into a SQLite candidate set."""
        fields = tuple(
            field
            for field in (
                "document_id",
                "document_name",
                "chapter",
                "section",
                "article",
                "clause",
                "point",
            )
            if getattr(metadata, field) is not None
        )
        if not self.enabled or not fields or metadata.confidence < self.min_confidence:
            return RetrievalFilter(fields=fields)
        ids = self.repository.filter_child_ids(metadata)
        return RetrievalFilter(
            candidate_ids=ids,
            applied=True,
            authoritative=metadata.document_id is not None,
            matched_count=len(ids),
            fields=fields,
        )

    def should_filter(self, query_metadata: QueryMetadata) -> bool:
        return self.build_filter(query_metadata).applied

    def allowed_ids(self, query_metadata: QueryMetadata) -> set[str] | None:
        result = self.build_filter(query_metadata)
        if not result.applied:
            return None
        if not result.candidate_ids and self.fallback_to_full_corpus:
            return None
        return result.candidate_ids
