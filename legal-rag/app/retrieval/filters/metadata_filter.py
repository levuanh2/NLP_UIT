"""Confidence-gated metadata filter."""

from app.domain.queries import QueryMetadata
from app.indexing.metadata_store.repository import LegalRepository


class MetadataFilter:
    def __init__(
        self,
        repository: LegalRepository,
        enabled: bool,
        min_confidence: float,
        fallback_to_full_corpus: bool,
    ) -> None:
        self.repository = repository
        self.enabled = enabled
        self.min_confidence = min_confidence
        self.fallback_to_full_corpus = fallback_to_full_corpus

    def should_filter(self, query_metadata: QueryMetadata) -> bool:
        """Return whether useful, sufficiently confident metadata may filter."""
        useful = any(
            (
                query_metadata.document_name,
                query_metadata.document_number,
                query_metadata.document_type,
                query_metadata.issued_year,
                query_metadata.article,
                query_metadata.clause,
            )
        )
        return (
            self.enabled and useful and query_metadata.confidence >= self.min_confidence
        )

    def allowed_ids(self, query_metadata: QueryMetadata) -> set[str] | None:
        """Return matching child IDs or full-corpus fallback (``None``)."""
        if not self.should_filter(query_metadata):
            return None
        identifiers = self.repository.filter_child_ids(query_metadata)
        if identifiers:
            return identifiers
        return None if self.fallback_to_full_corpus else set()
