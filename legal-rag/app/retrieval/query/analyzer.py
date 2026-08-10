"""Deterministic legal query analysis."""

from app.domain.queries import QueryMetadata
from app.retrieval.query.metadata_extractor import QueryMetadataExtractor


class QueryAnalyzer:
    def __init__(
        self, metadata_extractor: QueryMetadataExtractor | None = None
    ) -> None:
        self.metadata_extractor = metadata_extractor or QueryMetadataExtractor()

    def analyze(self, query: str) -> QueryMetadata:
        """Extract explicit metadata without guessing unspecified legal sources."""
        if not query or not query.strip():
            raise ValueError("Retrieval query must not be blank")
        return self.metadata_extractor.extract(query)
