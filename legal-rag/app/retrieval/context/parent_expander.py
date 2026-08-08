"""Parent context expansion."""

from app.domain.retrieval import LegalEvidence, RetrievalCandidate
from app.indexing.metadata_store.repository import LegalRepository


class ParentContextExpander:
    def __init__(self, repository: LegalRepository) -> None:
        self.repository = repository

    def expand(self, candidates: list[RetrievalCandidate]) -> list[LegalEvidence]:
        """Map children to parent chunks and deduplicate parents."""
        evidences: list[LegalEvidence] = []
        seen_parents: set[str] = set()
        for candidate in candidates:
            child = self.repository.get_child(candidate.child_id)
            if child is None or child.parent_id in seen_parents:
                continue
            parent = self.repository.get_parent(child.parent_id)
            if parent is None:
                continue
            seen_parents.add(child.parent_id)
            evidences.append(
                LegalEvidence(
                    evidence_id=child.parent_id,
                    document_id=parent.document_id,
                    document_name=child.metadata.document_name,
                    source_link=child.metadata.source_link,
                    chapter=parent.chapter,
                    section=parent.section,
                    article=parent.article,
                    clause=child.clause,
                    point=child.point,
                    text=parent.text,
                )
            )
        return evidences
