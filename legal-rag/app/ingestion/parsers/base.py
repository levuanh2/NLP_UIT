"""Document parser contract."""

from abc import ABC, abstractmethod
from pathlib import Path

from app.domain.documents import LegalDocument


class BaseDocumentParser(ABC):
    @abstractmethod
    def supports(self, source_path: Path) -> bool:
        """Return whether this parser supports the input file."""

    @abstractmethod
    def parse(self, source_path: Path) -> LegalDocument:
        """Parse one legal document."""
