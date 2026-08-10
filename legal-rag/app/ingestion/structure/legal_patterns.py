"""Centralized Vietnamese legal hierarchy patterns."""

CHAPTER_PATTERN = r"(?im)^\s*Chương\s+([IVXLCDM]+|\d+)\s*[:.]?\s*(.*)$"
SECTION_PATTERN = r"(?im)^\s*Mục\s+(\d+[A-Za-zĐđ]?)\s*[:.]?\s*(.*)$"
ARTICLE_PATTERN = r"(?im)^\s*Điều\s+(\d+[A-Za-zĐđ]?)\s*[.:]?\s*(.*)$"
CLAUSE_PATTERN = r"(?m)^\s*(\d+)\s*[.)]\s+(.+)$"
POINT_PATTERN = r"(?im)^\s*([a-zđ])\s*[.)]\s+(.+)$"
