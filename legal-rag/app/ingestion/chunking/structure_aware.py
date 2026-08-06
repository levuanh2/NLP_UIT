"""Structure-aware chunking helpers."""

from app.domain.documents import LegalDocument


class StructureAwareSegmenter:
    def segment(self, document: LegalDocument) -> list[str]:
        """Segment text without breaking explicit legal units."""
        # TODO(phase-implementation):
        # Segment the extracted hierarchy using configured token bounds.
        raise NotImplementedError
