"""Vietnamese legal structure extractor skeleton."""

from app.domain.documents import LegalDocument


class LegalStructureExtractor:
    def extract(self, document: LegalDocument) -> LegalDocument:
        """Populate chapter/section/article/clause/point hierarchy."""
        # TODO(phase-implementation):
        # Extract and attach validated Vietnamese legal hierarchy nodes.
        raise NotImplementedError
