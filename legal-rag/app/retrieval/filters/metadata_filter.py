"""Confidence-gated metadata filter skeleton."""

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
        # TODO(phase-implementation):
        # Implement confidence and useful-field validation.
        raise NotImplementedError

    def allowed_ids(self, query_metadata: QueryMetadata) -> set[str] | None:
        """Return matching child IDs or full-corpus fallback (``None``)."""
        # TODO(phase-implementation):
        # Apply pre-retrieval filtering and fall back when no IDs match.
        raise NotImplementedError
