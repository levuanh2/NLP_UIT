"""Legal query analysis skeleton."""

from app.domain.queries import QueryAnalysis
from app.retrieval.query.metadata_extractor import QueryMetadataExtractor


class QueryAnalyzer:
    def __init__(self, metadata_extractor: QueryMetadataExtractor) -> None:
        self.metadata_extractor = metadata_extractor

    def analyze(self, query: str) -> QueryAnalysis:
        # TODO(phase-implementation):
        # Normalize the query, detect intent, and extract metadata.
        raise NotImplementedError
