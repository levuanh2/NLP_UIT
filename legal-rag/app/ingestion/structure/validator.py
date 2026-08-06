"""Extracted hierarchy validation skeleton."""

from app.domain.documents import LegalDocument


class LegalStructureValidator:
    def validate(self, document: LegalDocument) -> list[str]:
        """Return hierarchy validation errors."""
        # TODO(phase-implementation):
        # Validate identifiers, ordering, and parent-child consistency.
        raise NotImplementedError
