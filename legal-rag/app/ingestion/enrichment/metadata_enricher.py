"""Legal metadata enrichment."""

from app.domain.chunks import ChildChunk
from app.domain.documents import LegalDocument


class MetadataEnricher:
    def enrich(
        self, document: LegalDocument, child_chunks: list[ChildChunk]
    ) -> list[ChildChunk]:
        """Attach document and hierarchy metadata to child chunks."""
        enriched: list[ChildChunk] = []
        for child in child_chunks:
            metadata = child.metadata.model_copy(
                update={
                    "document_id": document.document_id,
                    "document_name": document.document_name,
                    "source_link": document.source_link,
                }
            )
            enriched.append(
                child.model_copy(
                    update={
                        "document_id": document.document_id,
                        "metadata": metadata,
                    }
                )
            )
        return enriched
