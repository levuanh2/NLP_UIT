"""Hierarchy traversal skeleton."""

from app.domain.documents import LegalDocument


class HierarchicalChunkPlanner:
    def plan(self, document: LegalDocument) -> list[tuple[str, list[str]]]:
        """Plan parent units and their child legal segments."""
        # TODO(phase-implementation):
        # Produce stable hierarchy-aware chunk plans.
        raise NotImplementedError
