"""Structure-aware chunking helpers."""

from app.domain.documents import LegalDocument


class StructureAwareSegmenter:
    def segment(self, document: LegalDocument) -> list[str]:
        """Segment text without breaking explicit legal units."""
        if document.structure is None:
            raise ValueError("Document structure has not been extracted.")
        segments: list[str] = []
        for chapter in document.structure.chapters:
            for section in chapter.sections:
                for article in section.articles:
                    header = f"Điều {article.article_number}"
                    if article.title:
                        header += f". {article.title}"
                    for clause in article.clauses:
                        if clause.text:
                            segments.append(f"{header}\n{clause.text}")
        return segments
