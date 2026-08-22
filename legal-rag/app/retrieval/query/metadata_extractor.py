"""Deterministic extraction of explicit Vietnamese legal references."""

import re

from app.domain import legal_identifier
from app.domain.queries import QueryMetadata


class QueryMetadataExtractor:
    """Extract only references explicitly present in a query."""

    _ARTICLE = re.compile(r"(?i)(?<!\w)điều\s+(\d+[a-zđ]?)\b")
    _CLAUSE = re.compile(r"(?i)(?<!\w)khoản\s+(\d+[a-zđ]?)\b")
    _POINT = re.compile(r"(?i)(?<!\w)điểm\s+([a-zđ]|\d+)\b")
    _CHAPTER = re.compile(r"(?i)(?<!\w)chương\s+([ivxlcdm]+|\d+)\b")
    _SECTION = re.compile(r"(?i)(?<!\w)mục\s+(\d+[a-zđ]?)\b")
    _DOCUMENT_ID = re.compile(
        r"(?i)(?:document_id|document\s+id|văn\s+bản\s+id)\s*[:=#]?\s*(\d+)\b"
    )
    _DOCUMENT_NAME = re.compile(
        r"(?i)\b((?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|quyết\s+định|"
        r"pháp\s+lệnh)\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ]*){0,7})"
    )

    def extract(self, query: str) -> QueryMetadata:
        raw_query = query.strip()
        article = self._reference(self._ARTICLE, raw_query, "Điều")
        clause = self._reference(self._CLAUSE, raw_query, "Khoản")
        point = self._reference(self._POINT, raw_query, "Điểm", lower=True)
        chapter = self._reference(self._CHAPTER, raw_query, "Chương")
        section = self._reference(self._SECTION, raw_query, "Mục")
        document_id_match = self._DOCUMENT_ID.search(raw_query)
        document_name_match = self._DOCUMENT_NAME.search(raw_query)
        document_name = (
            self._clean_document_name(document_name_match.group(1))
            if document_name_match
            else None
        )
        document_number = legal_identifier.find(raw_query)
        # Confidence gates the pre-retrieval filter, so only a reference that
        # identifies one document earns it. A bare "Luật" or "quyết định" used
        # to reach 1.0 through document_name and filtered on a phrase lifted out
        # of ordinary prose; measured on 200 dev questions that fired 11 times
        # and matched nothing 11 times. A wrong filter silently removes the
        # right evidence, so anything short of a structured identifier now
        # leaves retrieval unfiltered.
        authoritative = document_number is not None or document_id_match is not None
        return QueryMetadata(
            raw_query=raw_query,
            document_name=document_name,
            document_number=document_number,
            document_id=(
                int(document_id_match.group(1)) if document_id_match else None
            ),
            chapter=chapter,
            section=section,
            article=article,
            clause=clause,
            point=point,
            confidence=1.0 if authoritative else 0.0,
        )

    @staticmethod
    def _reference(
        pattern: re.Pattern[str], query: str, label: str, *, lower: bool = False
    ) -> str | None:
        match = pattern.search(query)
        if not match:
            return None
        value = match.group(1).lower() if lower else match.group(1).upper()
        return f"{label} {value}"

    @staticmethod
    def _clean_document_name(value: str) -> str:
        normalized = re.sub(r"\s+", " ", value).strip(" ,.;:?!…")
        # Stop at ordinary predicate words so a title is not guessed from the
        # remainder of a sentence such as "Luật Doanh nghiệp quy định ...".
        stop = re.search(
            r"(?i)\s+(?:quy\s+định|được|phải|có|là|thì|tại|theo|nêu)\b",
            normalized,
        )
        return normalized[: stop.start()].strip() if stop else normalized
