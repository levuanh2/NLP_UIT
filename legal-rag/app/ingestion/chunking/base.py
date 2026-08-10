"""Chunker contract."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument


class BaseChunker(ABC):
    @abstractmethod
    def iter_chunk_groups(
        self, document: LegalDocument
    ) -> Iterator[tuple[ParentChunk, list[ChildChunk]]]:
        """Yield one bounded parent and its children at a time."""

    @abstractmethod
    def chunk(
        self, document: LegalDocument
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Create parent and child chunks."""
