"""Rule-assisted Vietnamese legal reference extraction."""

import re

from app.domain.queries import QueryMetadata


class QueryMetadataExtractor:
    def extract(self, query: str) -> QueryMetadata:
        article = re.search(r"(?i)\bđiều\s+(\d+[a-zđ]?)", query)
        clause = re.search(r"(?i)\bkhoản\s+(\d+)", query)
        document = re.search(
            r"(?i)\b(nghị định|thông tư|luật|quyết định|nghị quyết)\s+"
            r"(?:số\s+)?([\d/.-]+(?:nđ-cp|tt-[a-zđ]+|qđ-[a-zđ]+)?)",
            query,
        )
        year = re.search(r"\b(19\d{2}|20\d{2})\b", query)
        useful = sum(bool(item) for item in (article, clause, document, year))
        confidence = min(1.0, 0.55 + 0.15 * useful) if useful else 0.0
        return QueryMetadata(
            document_name=document.group(0).strip() if document else None,
            document_number=document.group(2).strip() if document else None,
            document_type=document.group(1).casefold() if document else None,
            issued_year=int(year.group(1)) if year else None,
            article=article.group(1).casefold() if article else None,
            clause=clause.group(1) if clause else None,
            legal_topic=None,
            confidence=confidence,
        )
