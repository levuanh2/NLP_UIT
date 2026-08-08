"""Legal hierarchy traversal."""

from app.domain.documents import LegalDocument


class HierarchicalChunkPlanner:
    def plan(self, document: LegalDocument) -> list[tuple[str, list[str]]]:
        """Plan parent units and their child legal segments."""
        if document.structure is None:
            raise ValueError("Document structure has not been extracted.")
        plans: list[tuple[str, list[str]]] = []
        for chapter in document.structure.chapters:
            for section in chapter.sections:
                for article in section.articles:
                    segments = [
                        clause.text for clause in article.clauses if clause.text
                    ]
                    plans.append((article.article_id, segments))
        return plans
