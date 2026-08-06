"""Chunker contract."""

from abc import ABC, abstractmethod

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument


class BaseChunker(ABC):
    @abstractmethod
    def chunk(
        self, document: LegalDocument
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Create parent and child chunks."""
