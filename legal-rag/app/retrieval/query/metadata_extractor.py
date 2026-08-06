"""Rule/model-assisted query metadata extraction skeleton."""

from app.domain.queries import QueryMetadata


class QueryMetadataExtractor:
    def extract(self, query: str) -> QueryMetadata:
        # TODO(phase-implementation):
        # Extract explicit Vietnamese legal references and calibrated confidence.
        raise NotImplementedError
