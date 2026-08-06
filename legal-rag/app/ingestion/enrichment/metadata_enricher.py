"""Legal metadata enrichment skeleton."""

from app.domain.chunks import ChildChunk
from app.domain.documents import LegalDocument


class MetadataEnricher:
    def enrich(
        self, document: LegalDocument, child_chunks: list[ChildChunk]
    ) -> list[ChildChunk]:
        """Attach document and hierarchy metadata to child chunks."""
        # TODO(phase-implementation):
        # Merge extracted document metadata into every child chunk.
        raise NotImplementedError
