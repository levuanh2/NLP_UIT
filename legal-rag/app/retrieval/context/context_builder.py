"""Legal evidence context builder."""

import re

from app.domain.retrieval import LegalContext, LegalEvidence


class LegalContextBuilder:
    def __init__(self, max_context_words: int = 1200) -> None:
        self.max_context_words = max_context_words

    def build(self, query: str, evidences: list[LegalEvidence]) -> LegalContext:
        """Format legal evidence for the LLM."""
        sections: list[str] = []
        for index, evidence in enumerate(evidences, start=1):
            location = [
                value
                for value in (
                    f"Điều {evidence.article}" if evidence.article else None,
                    f"khoản {evidence.clause}" if evidence.clause else None,
                    f"điểm {evidence.point}" if evidence.point else None,
                )
                if value
            ]
            sections.append(
                "\n".join(
                    (
                        f"[E{index}] evidence_id={evidence.evidence_id}",
                        f"Văn bản: {_display_document_name(evidence.document_name)}",
                        f"Vị trí: {', '.join(location) or 'Không rõ'}",
                        f"Nguồn: {evidence.source_link or 'Không rõ'}",
                        evidence.text,
                    )
                )
            )
        formatted = "\n\n".join(sections)
        words = formatted.split()
        if len(words) > self.max_context_words:
            formatted = " ".join(words[: self.max_context_words])
        return LegalContext(
            query=query,
            evidences=evidences,
            formatted_context=formatted,
            token_count=len(formatted.split()),
        )


_LEGAL_SLUG_PATTERNS = (
    (
        re.compile(r"(?i)nghi-dinh-(\d+[a-z]?)-(\d{4})-nd-cp"),
        lambda match: f"Nghị định {match[1]}/{match[2]}/NĐ-CP",
    ),
    (
        re.compile(r"(?i)thong-tu-(\d+[a-z]?)-(\d{4})-tt-([a-z0-9]+)"),
        lambda match: f"Thông tư {match[1]}/{match[2]}/TT-{match[3].upper()}",
    ),
    (
        re.compile(r"(?i)quyet-dinh-(\d+[a-z]?)-(\d{4})-qd-([a-z0-9]+)"),
        lambda match: f"Quyết định {match[1]}/{match[2]}/QĐ-{match[3].upper()}",
    ),
)


def _display_document_name(name: str) -> str:
    if not name:
        return "Không rõ"
    for pattern, formatter in _LEGAL_SLUG_PATTERNS:
        match = pattern.search(name)
        if match:
            return formatter(match)
    return name
