"""Extracted hierarchy validation."""

from app.domain.documents import LegalDocument


class LegalStructureValidator:
    def validate(self, document: LegalDocument) -> list[str]:
        """Return hierarchy validation errors."""
        if document.structure is None:
            return ["Document has no extracted legal structure."]
        errors: list[str] = []
        identifiers: set[str] = set()
        for chapter in document.structure.chapters:
            nodes = [chapter.chapter_id]
            for section in chapter.sections:
                nodes.append(section.section_id)
                for article in section.articles:
                    nodes.append(article.article_id)
                    if not article.clauses:
                        errors.append(f"Article has no clauses: {article.article_id}")
                    nodes.extend(clause.clause_id for clause in article.clauses)
            for identifier in nodes:
                if identifier in identifiers:
                    errors.append(f"Duplicate hierarchy ID: {identifier}")
                identifiers.add(identifier)
        return errors
