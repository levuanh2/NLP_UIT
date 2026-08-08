"""Conservative Vietnamese legal hierarchy extractor."""

import re

from app.domain.documents import (
    LegalArticle,
    LegalChapter,
    LegalClause,
    LegalDocument,
    LegalPoint,
    LegalSection,
    LegalStructure,
)
from app.ingestion.structure.legal_patterns import (
    ARTICLE_PATTERN,
    CHAPTER_PATTERN,
    CLAUSE_PATTERN,
    POINT_PATTERN,
    SECTION_PATTERN,
)


class LegalStructureExtractor:
    def extract(self, document: LegalDocument) -> LegalDocument:
        """Populate a lossless article/clause/point hierarchy."""
        text = document.cleaned_text or document.raw_text
        article_matches = list(re.finditer(ARTICLE_PATTERN, text))
        positioned_articles: list[tuple[int, LegalArticle]] = []
        if not article_matches:
            positioned_articles.append(
                (0, self._article(document.document_id, "0", 0, None, text))
            )
        else:
            introduction = text[: article_matches[0].start()].strip()
            if introduction:
                positioned_articles.append(
                    (
                        0,
                        self._article(
                            document.document_id,
                            "0",
                            0,
                            "Phần mở đầu",
                            introduction,
                        ),
                    )
                )
            for index, match in enumerate(article_matches):
                end = (
                    article_matches[index + 1].start()
                    if index + 1 < len(article_matches)
                    else len(text)
                )
                segment = text[match.start() : end].strip()
                first_line, separator, body = segment.partition("\n")
                marker_end = match.end() - match.start()
                title = first_line[marker_end:].strip(" .:-") or None
                article_body = body.strip() if separator else segment
                positioned_articles.append(
                    (
                        match.start(),
                        self._article(
                            document.document_id,
                            match.group(1),
                            index + 1,
                            title,
                            article_body,
                        ),
                    )
                )
        chapter_matches = list(re.finditer(CHAPTER_PATTERN, text))
        section_matches = list(re.finditer(SECTION_PATTERN, text))
        grouped: dict[tuple[str, str], list[LegalArticle]] = {}
        for position, article in positioned_articles:
            chapter_title = _active_heading(chapter_matches, position, "Toàn văn")
            section_title = _active_heading(section_matches, position, "Nội dung")
            grouped.setdefault((chapter_title, section_title), []).append(article)
        chapter_groups: dict[str, list[tuple[str, list[LegalArticle]]]] = {}
        for (chapter_title, section_title), articles in grouped.items():
            chapter_groups.setdefault(chapter_title, []).append(
                (section_title, articles)
            )
        chapters: list[LegalChapter] = []
        for chapter_index, (chapter_title, sections) in enumerate(
            chapter_groups.items()
        ):
            section_nodes = [
                LegalSection(
                    section_id=(
                        f"doc-{document.document_id}:chapter-{chapter_index}:"
                        f"section-{section_index}"
                    ),
                    title=section_title,
                    articles=articles,
                )
                for section_index, (section_title, articles) in enumerate(sections)
            ]
            chapters.append(
                LegalChapter(
                    chapter_id=f"doc-{document.document_id}:chapter-{chapter_index}",
                    title=chapter_title,
                    sections=section_nodes,
                )
            )
        return document.model_copy(
            update={"structure": LegalStructure(chapters=chapters)}
        )

    @staticmethod
    def _article(
        document_id: int,
        number: str,
        ordinal: int,
        title: str | None,
        body: str,
    ) -> LegalArticle:
        matches = list(re.finditer(CLAUSE_PATTERN, body))
        clauses: list[LegalClause] = []
        prefix = body[: matches[0].start()].strip() if matches else body.strip()
        if prefix or not matches:
            clauses.append(
                LegalStructureExtractor._clause(
                    document_id, number, ordinal, "0", prefix
                )
            )
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            clauses.append(
                LegalStructureExtractor._clause(
                    document_id,
                    number,
                    ordinal,
                    match.group(1),
                    body[match.start() : end].strip(),
                )
            )
        return LegalArticle(
            article_id=(
                f"doc-{document_id}:article-{number.casefold()}:ordinal-{ordinal}"
            ),
            article_number=number,
            title=title,
            clauses=clauses,
        )

    @staticmethod
    def _clause(
        document_id: int,
        article_number: str,
        article_ordinal: int,
        clause_number: str,
        text: str,
    ) -> LegalClause:
        matches = list(re.finditer(POINT_PATTERN, text))
        points: list[LegalPoint] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            label = match.group(1).casefold()
            points.append(
                LegalPoint(
                    point_id=(
                        f"doc-{document_id}:article-{article_number.casefold()}:"
                        f"ordinal-{article_ordinal}:clause-{clause_number}:point-{label}"
                    ),
                    point_label=label,
                    text=text[match.start() : end].strip(),
                )
            )
        return LegalClause(
            clause_id=(
                f"doc-{document_id}:article-{article_number.casefold()}:"
                f"ordinal-{article_ordinal}:clause-{clause_number}"
            ),
            clause_number=clause_number,
            text=text.strip(),
            points=points,
        )


def _active_heading(matches: list[re.Match[str]], position: int, fallback: str) -> str:
    active = [match for match in matches if match.start() <= position]
    return active[-1].group(0).strip() if active else fallback
