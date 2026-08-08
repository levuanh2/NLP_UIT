"""Centralized Vietnamese legal structure patterns."""

CHAPTER_PATTERN = r"(?im)^\s*CHƯƠNG\s+([IVXLCDM]+|\d+)\b"
SECTION_PATTERN = r"(?im)^\s*MỤC\s+(\d+)\b"
ARTICLE_PATTERN = r"(?im)^\s*ĐIỀU\s+(\d+[a-zA-ZĐđ]?)\b"
CLAUSE_PATTERN = r"(?m)^\s*(\d+)\.\s+"
POINT_PATTERN = r"(?im)^\s*([a-zđ])\)\s+"
