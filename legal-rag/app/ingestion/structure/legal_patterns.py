"""Centralized Vietnamese legal structure patterns."""

CHAPTER_PATTERN = r"(?im)^\s*CHƯƠNG\s+([IVXLCDM]+)\b"
SECTION_PATTERN = r"(?im)^\s*MỤC\s+(\d+)\b"
ARTICLE_PATTERN = r"(?im)^\s*ĐIỀU\s+(\d+[a-zA-Z]?)\b"
CLAUSE_PATTERN = r"(?m)^\s*(\d+)\.\s+"
POINT_PATTERN = r"(?m)^\s*([a-zđ])\)\s+"
