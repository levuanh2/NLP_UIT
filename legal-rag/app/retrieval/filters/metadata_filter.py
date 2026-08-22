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
    # Why a filter was declined, so a benchmark can count the reasons apart.
    ambiguous: bool = False
    empty_lookup: bool = False


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

    # document_name is extracted for diagnostics but never filtered on: it is a
    # phrase lifted out of prose ("quyết định hình phạt"), while the stored name
    # is a URL slug, so it can only ever match by accident.
    FILTERABLE = ("document_id", "document_number", "chapter", "section", "article",
                  "clause", "point")

    def build_filter(self, metadata: QueryMetadata) -> RetrievalFilter:
        """Convert only explicit metadata into a SQLite candidate set.

        Fail-safe throughout: a filter that matches nothing, or an identifier
        that resolves to more than one document, returns no filter at all. A
        false positive here deletes the right evidence before ranking ever sees
        it, which is strictly worse than searching the whole corpus.
        """
        fields = tuple(
            field for field in self.FILTERABLE if getattr(metadata, field) is not None
        )
        if not self.enabled or not fields or metadata.confidence < self.min_confidence:
            return RetrievalFilter(fields=fields)

        ambiguous = False
        if metadata.document_number and metadata.document_id is None:
            documents = self.repository.document_ids_for_identifier(
                metadata.document_number
            )
            if len(documents) != 1:
                ambiguous = True

        if ambiguous:
            return RetrievalFilter(fields=fields, ambiguous=True)

        ids = self.repository.filter_child_ids(metadata)
        if not ids:
            return RetrievalFilter(fields=fields, empty_lookup=True)
        return RetrievalFilter(
            candidate_ids=ids,
            applied=True,
            authoritative=(
                metadata.document_id is not None or metadata.document_number is not None
            ),
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
