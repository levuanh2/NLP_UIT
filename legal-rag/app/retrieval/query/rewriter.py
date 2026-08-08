"""Deterministic Vietnamese legal query rewriting."""

import re
import unicodedata


class QueryRewriter:
    def rewrite(self, query: str) -> str:
        normalized = unicodedata.normalize("NFC", query).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        replacements = {
            r"(?i)\bNĐ-CP\b": "Nghị định",
            r"(?i)\bTT-BTC\b": "Thông tư Bộ Tài chính",
            r"(?i)\bQĐ\b": "Quyết định",
        }
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized)
        return normalized
