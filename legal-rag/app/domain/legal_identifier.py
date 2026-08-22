"""Canonicalisation of Vietnamese legal document identifiers.

Two different normalisations are needed and they must not be confused:

- the **canonical identifier** is what a person writes and what the corpus
  means: ``17/2022/TT-BGTVT``, ``569/QĐ-TTg``. Accents are part of it (``QĐ``
  is not ``QD``), so it is never ASCII-folded.
- the **slug fragment** is how that identifier appears inside the stored
  ``document_name``, which is a URL slug: ``Thong-tu-17-2022-TT-BGTVT-...``.
  There the separators are hyphens and the accents are already gone.

Folding the first into the second loses legal information, so the fold happens
once, at lookup time, and only in that direction.
"""

import re
import unicodedata

# Two shapes actually present in the corpus, counted over 7407 distinct
# document names: 4799 carry a number/year/type ("17-2022-TT-BGTVT"), and the
# remainder carry a number/type with the year elsewhere ("569-QD-TTg-2022").
_TYPE = r"[A-Za-zĐĐ][A-Za-z0-9ĐĐ]*(?:-[A-Za-z0-9ĐĐ]+)*"
NUMBER_YEAR_TYPE = re.compile(
    rf"(?<![\d/-])(\d{{1,5}})\s*[/-]\s*(\d{{4}})\s*[/-]\s*({_TYPE})"
)
NUMBER_TYPE = re.compile(
    rf"(?<![\d/-])(\d{{1,5}})\s*[/-]\s*"
    rf"((?:Q[ĐD]|N[ĐD]|TT|TTLT|CT|NQ|PL|QH\d*|L|SL|UBTVQH\d*)(?:-{_TYPE})?)",
    re.IGNORECASE,
)


def find(text: str) -> str | None:
    """The first structured legal identifier in the text, canonicalised."""
    match = NUMBER_YEAR_TYPE.search(text)
    if match:
        number, year, kind = match.groups()
        return f"{number}/{year}/{_tidy(kind)}"
    match = NUMBER_TYPE.search(text)
    if match:
        number, kind = match.groups()
        return f"{number}/{_tidy(kind)}"
    return None


def _tidy(kind: str) -> str:
    """Uppercase the acronym part while leaving mixed-case agencies alone.

    "TTg" in ``569/QĐ-TTg`` is written that way in the corpus, so only segments
    that are already all one case are safe to uppercase.
    """
    segments = []
    for segment in kind.split("-"):
        segments.append(segment.upper() if segment.islower() else segment)
    return "-".join(segments)


def slug_fragment(identifier: str) -> str:
    """The identifier as it appears inside a stored document-name slug."""
    folded = unicodedata.normalize("NFD", identifier)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    folded = folded.replace("Đ", "D").replace("đ", "d")
    return folded.replace("/", "-")


def escape_like(value: str) -> str:
    """Escape LIKE wildcards so an identifier cannot act as a pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
