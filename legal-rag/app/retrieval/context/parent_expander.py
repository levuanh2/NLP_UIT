"""Parent context expansion skeleton."""

from app.domain.retrieval import LegalEvidence, RetrievalCandidate
from app.indexing.metadata_store.repository import LegalRepository


class ParentContextExpander:
    def __init__(self, repository: LegalRepository) -> None:
        self.repository = repository

    def expand(
        self, candidates: list[RetrievalCandidate]
    ) -> list[LegalEvidence]:
        """Map children to parent chunks and deduplicate parents."""
        # TODO(phase-implementation):
        # Preserve rank while mapping children to unique parent evidence.
        raise NotImplementedError
