"""Deterministic validation and resolution of numeric evidence citations."""

import re

from app.domain.generation import Citation, CitationValidationResult, GeneratedAnswer
from app.domain.retrieval import LegalContext, LegalEvidence, RetrievalResult


class CitationValidator:
    _REFERENCE = re.compile(r"\[(\d+)]")

    def __init__(self, require_citation: bool = True) -> None:
        self.require_citation = require_citation

    def validate(
        self,
        answer: str | GeneratedAnswer,
        source: RetrievalResult | LegalContext,
        *,
        evidences: list[LegalEvidence] | None = None,
    ) -> CitationValidationResult:
        text = answer.answer if isinstance(answer, GeneratedAnswer) else answer
        available = evidences if evidences is not None else source.evidences
        citation_ids = list(
            dict.fromkeys(int(value) for value in self._REFERENCE.findall(text))
        )
        errors: list[str] = []
        if (
            not citation_ids
            and self.require_citation
            and not self.is_safe_abstention(text)
        ):
            errors.append("Answer contains no evidence citation.")
        resolved: list[Citation] = []
        for citation_id in citation_ids:
            if citation_id < 1 or citation_id > len(available):
                errors.append(f"Citation [{citation_id}] does not exist in context.")
                continue
            evidence = available[citation_id - 1]
            resolved.append(self._citation(evidence))
        return CitationValidationResult(
            valid=not errors,
            citation_ids=citation_ids,
            citations=resolved,
            errors=errors,
        )

    @staticmethod
    def is_safe_abstention(answer: str) -> bool:
        normalized = " ".join(answer.lower().split())
        return (
            "không tìm thấy đủ căn cứ" in normalized
            or "tài liệu được cung cấp chưa đủ căn cứ" in normalized
            or "không đủ căn cứ trong tài liệu" in normalized
        )

    @staticmethod
    def _citation(evidence: LegalEvidence) -> Citation:
        return Citation(
            document_id=evidence.document_id,
            document_name=evidence.document_name,
            source_link=evidence.source_link,
            chapter=evidence.chapter,
            article=evidence.article,
            clause=evidence.clause,
            point=evidence.point,
            child_id=evidence.child_id,
            evidence_id=evidence.evidence_id,
        )
