"""Expand ranked child hits to same-parent neighbor context."""

from dataclasses import dataclass, field

from app.domain.chunks import ChildChunk
from app.domain.retrieval import LegalEvidence, RetrievalCandidate
from app.indexing.metadata_store.repository import LegalRepository


@dataclass
class _ParentSelection:
    """Child union for one selected parent, anchored by its best direct hit."""

    anchor: ChildChunk
    rank: int
    children: dict[str, ChildChunk] = field(default_factory=dict)


class ParentContextExpander:
    def __init__(
        self,
        repository: LegalRepository,
        neighbor_window: int = 1,
        same_parent_only: bool = True,
        deduplicate_overlap: bool = True,
        max_parents_per_document: int = 3,
    ) -> None:
        self.repository = repository
        self.neighbor_window = neighbor_window
        self.same_parent_only = same_parent_only
        self.deduplicate_overlap = deduplicate_overlap
        self.max_parents_per_document = max_parents_per_document

    def expand(self, candidates: list[RetrievalCandidate]) -> list[LegalEvidence]:
        """Union every direct hit and its same-parent neighbors before grouping.

        Parent-level deduplication is deliberately deferred until all candidates
        have contributed children. Otherwise, a later direct hit from an already
        selected parent can be discarded even though its neighbor window reaches
        legal clauses that the first hit did not cover.
        """
        selections: dict[str, _ParentSelection] = {}
        parents_per_document: dict[int, int] = {}
        for rank, candidate in enumerate(candidates, start=1):
            child = self.repository.get_child(candidate.child_id)
            if child is None:
                continue

            selection = selections.get(child.parent_id)
            if selection is None:
                parent_count = parents_per_document.get(child.document_id, 0)
                if parent_count >= self.max_parents_per_document:
                    continue
                if self.repository.get_parent(child.parent_id) is None:
                    continue
                selection = _ParentSelection(anchor=child, rank=rank)
                selections[child.parent_id] = selection
                parents_per_document[child.document_id] = parent_count + 1

            # A direct reranker hit is authoritative evidence for expansion and
            # must survive even if a repository implementation omits the current
            # child from its neighbor query.
            selection.children[child.child_id] = child
            neighbors = self.repository.get_neighbor_children(
                child.child_id, self.neighbor_window
            )
            # This is a retrieval safety boundary, not a presentation option:
            # expansion must never leak into another legal parent/document even
            # if a repository returns a wider positional window. Keep the
            # constructor flag for compatibility with existing callers.
            neighbors = [
                item
                for item in neighbors
                if item.parent_id == child.parent_id
                and item.document_id == child.document_id
            ]
            for neighbor in neighbors:
                selection.children.setdefault(neighbor.child_id, neighbor)

        evidences: list[LegalEvidence] = []
        # Parent groups remain ordered by their best direct reranker hit. This is
        # the priority consumed by ContextBudgetManager; neighbor-only chunks never
        # promote an otherwise lower-ranked parent.
        for parent_id, selection in sorted(
            selections.items(), key=lambda item: (item[1].rank, item[0])
        ):
            anchor = selection.anchor
            children = sorted(
                selection.children.values(),
                key=lambda item: (
                    item.document_id,
                    item.parent_id,
                    item.position,
                    item.child_id,
                ),
            )
            text = self._join_neighbors(children)
            evidences.append(
                LegalEvidence(
                    evidence_id=parent_id,
                    parent_id=parent_id,
                    child_id=anchor.child_id,
                    document_id=anchor.document_id,
                    document_name=anchor.document_name,
                    source_link=anchor.source_link,
                    chapter=anchor.chapter,
                    section=anchor.section,
                    article=anchor.article,
                    clause=anchor.clause,
                    point=anchor.point,
                    position=anchor.position,
                    rank=selection.rank,
                    text=text,
                )
            )
        return evidences

    def _join_neighbors(self, children: list[ChildChunk]) -> str:
        if not children:
            return ""
        result = children[0].text
        for child in children[1:]:
            if self.deduplicate_overlap:
                result = self._merge_overlap(result, child.text)
            else:
                result = f"{result}\n\n{child.text}"
        return result

    @staticmethod
    def _merge_overlap(left: str, right: str, max_words: int = 64) -> str:
        left_words = left.split()
        right_words = right.split()
        overlap = 0
        for size in range(min(max_words, len(left_words), len(right_words)), 0, -1):
            if left_words[-size:] == right_words[:size]:
                overlap = size
                break
        suffix = " ".join(right_words[overlap:])
        return f"{left}\n\n{suffix}".rstrip()
