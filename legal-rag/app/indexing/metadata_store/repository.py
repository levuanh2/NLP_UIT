"""Legal metadata repository skeleton."""

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.metadata import LegalMetadata
from app.domain.queries import QueryMetadata


class LegalRepository:
    def save_chunks(
        self, parent_chunks: list[ParentChunk], child_chunks: list[ChildChunk]
    ) -> None:
        # TODO(phase-implementation):
        # Persist chunk text and explicit JSON-derived/hierarchy metadata
        # columns transactionally. FAISS must contain vectors only.
        raise NotImplementedError

    def get_child(self, child_id: str) -> ChildChunk | None:
        # TODO(phase-implementation):
        # Load and map one persisted child chunk.
        raise NotImplementedError

    def get_parent(self, parent_id: str) -> ParentChunk | None:
        # TODO(phase-implementation):
        # Load and map one persisted parent chunk.
        raise NotImplementedError

    def filter_child_ids(self, metadata: QueryMetadata) -> set[str]:
        # TODO(phase-implementation):
        # Translate confident query metadata into safe SQLite predicates.
        raise NotImplementedError

    def get_metadata(self, child_id: str) -> LegalMetadata | None:
        # TODO(phase-implementation):
        # Load typed legal metadata for a child chunk.
        raise NotImplementedError
