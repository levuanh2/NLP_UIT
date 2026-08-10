"""Regex-based Vietnamese legal hierarchy extraction."""

import re
from typing import overload

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
    @overload
    def extract(self, value: str) -> LegalStructure: ...

    @overload
    def extract(self, value: LegalDocument) -> LegalDocument: ...

    def extract(self, value: str | LegalDocument) -> LegalStructure | LegalDocument:
        """Detect hierarchy without requiring every legal level to exist."""
        text = (
            value if isinstance(value, str) else (value.cleaned_text or value.raw_text)
        )
        structure = self._extract_structure(text)
        if isinstance(value, str):
            return structure
        return value.model_copy(update={"structure": structure})

    def _extract_structure(self, text: str) -> LegalStructure:
        article_matches = list(re.finditer(ARTICLE_PATTERN, text))
        if not article_matches:
            return LegalStructure()

        chapter_matches = list(re.finditer(CHAPTER_PATTERN, text))
        section_matches = list(re.finditer(SECTION_PATTERN, text))
        chapters: dict[str, LegalChapter] = {}
        root_sections: dict[str, LegalSection] = {}
        root_articles: list[LegalArticle] = []

        for position, match in enumerate(article_matches):
            end = (
                article_matches[position + 1].start()
                if position + 1 < len(article_matches)
                else len(text)
            )
            article_text = text[match.start() : end].strip()
            number = match.group(1)
            article = LegalArticle(
                article_id=f"article:{number}:{position}",
                article_number=number,
                title=match.group(2).strip() or None,
                clauses=self._extract_clauses(article_text),
                text=article_text,
                position=position,
            )
            chapter_match = self._last_before(chapter_matches, match.start())
            section_match = self._last_before(section_matches, match.start())
            chapter_label = self._heading_label("Chương", chapter_match)
            section_label = self._heading_label("Mục", section_match)

            if chapter_label:
                chapter = chapters.setdefault(
                    chapter_label,
                    LegalChapter(
                        chapter_id=f"chapter:{len(chapters)}",
                        title=chapter_label,
                        position=len(chapters),
                    ),
                )
                section_belongs = section_match is not None and (
                    chapter_match is None
                    or section_match.start() > chapter_match.start()
                )
                if section_label and section_belongs:
                    section = next(
                        (
                            item
                            for item in chapter.sections
                            if item.title == section_label
                        ),
                        None,
                    )
                    if section is None:
                        section = LegalSection(
                            section_id=f"{chapter.chapter_id}:section:{len(chapter.sections)}",
                            title=section_label,
                            position=len(chapter.sections),
                        )
                        chapter.sections.append(section)
                    section.articles.append(article)
                else:
                    chapter.articles.append(article)
            elif section_label:
                section = root_sections.setdefault(
                    section_label,
                    LegalSection(
                        section_id=f"section:{len(root_sections)}",
                        title=section_label,
                        position=len(root_sections),
                    ),
                )
                section.articles.append(article)
            else:
                root_articles.append(article)

        return LegalStructure(
            chapters=list(chapters.values()),
            sections=list(root_sections.values()),
            articles=root_articles,
        )

    def _extract_clauses(self, article_text: str) -> list[LegalClause]:
        matches = list(re.finditer(CLAUSE_PATTERN, article_text))
        clauses: list[LegalClause] = []
        for position, match in enumerate(matches):
            end = (
                matches[position + 1].start()
                if position + 1 < len(matches)
                else len(article_text)
            )
            clause_text = article_text[match.start() : end].strip()
            clauses.append(
                LegalClause(
                    clause_id=f"clause:{match.group(1)}:{position}",
                    clause_number=match.group(1),
                    text=clause_text,
                    points=self._extract_points(clause_text),
                    position=position,
                )
            )
        return clauses

    @staticmethod
    def _extract_points(clause_text: str) -> list[LegalPoint]:
        matches = list(re.finditer(POINT_PATTERN, clause_text))
        points: list[LegalPoint] = []
        for position, match in enumerate(matches):
            end = (
                matches[position + 1].start()
                if position + 1 < len(matches)
                else len(clause_text)
            )
            points.append(
                LegalPoint(
                    point_id=f"point:{match.group(1)}:{position}",
                    point_label=match.group(1),
                    text=clause_text[match.start() : end].strip(),
                    position=position,
                )
            )
        return points

    @staticmethod
    def _last_before(matches: list[re.Match[str]], offset: int) -> re.Match[str] | None:
        result = None
        for match in matches:
            if match.start() >= offset:
                break
            result = match
        return result

    @staticmethod
    def _heading_label(prefix: str, match: re.Match[str] | None) -> str | None:
        if match is None:
            return None
        suffix = (
            match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
        )
        return " ".join(part for part in (prefix, match.group(1), suffix) if part)
