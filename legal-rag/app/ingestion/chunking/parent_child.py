"""Parent-child legal chunker skeleton."""

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument
from app.ingestion.chunking.base import BaseChunker


class ParentChildChunker(BaseChunker):
    def __init__(
        self,
        parent_target_tokens: int,
        parent_max_tokens: int,
        child_target_tokens: int,
        child_max_tokens: int,
    ) -> None:
        self.parent_target_tokens = parent_target_tokens
        self.parent_max_tokens = parent_max_tokens
        self.child_target_tokens = child_target_tokens
        self.child_max_tokens = child_max_tokens

    def chunk(
        self, document: LegalDocument
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        # TODO(phase-implementation):
        # Create stable article parents and clause/semantic child chunks.
        raise NotImplementedError
