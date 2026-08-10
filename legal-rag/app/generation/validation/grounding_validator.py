"""Conservative deterministic grounding checks; no LLM judge."""

import re
import unicodedata

from app.domain.generation import (
    CitationValidationResult,
    GeneratedAnswer,
    GroundingResult,
)
from app.domain.retrieval import LegalContext, LegalEvidence, RetrievalResult
from app.generation.validation.citation_validator import CitationValidator


class GroundingValidator:
    _CITATION = re.compile(r"\[(\d+)]")
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
    _LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
    _MARKDOWN = re.compile(r"[*_`#]+")
    _TOKEN = re.compile(r"[a-z0-9]+")
    _STOP_WORDS = {
        "ai",
        "bi",
        "boi",
        "cac",
        "can",
        "cho",
        "co",
        "cua",
        "cung",
        "da",
        "de",
        "do",
        "duoc",
        "hay",
        "khi",
        "la",
        "ma",
        "mot",
        "nay",
        "nhung",
        "nhu",
        "phai",
        "qua",
        "rang",
        "se",
        "sau",
        "tai",
        "theo",
        "thi",
        "tren",
        "tu",
        "va",
        "ve",
        "viec",
        "voi",
    }
    _DETAIL_PATTERNS = (
        re.compile(r"\btrong \d+ (?:ngay|thang|nam)\b"),
        re.compile(r"\btruoc khi\b"),
        re.compile(r"\bsau khi\b"),
        re.compile(r"\bco hieu luc\b"),
        re.compile(r"\bchi khi\b"),
        re.compile(r"\bngoai tru\b"),
        re.compile(r"\bkhong duoc\b"),
        re.compile(r"\bduoc phep\b"),
    )
    _ALL_EMPHASIS = re.compile(
        r"\btat ca cac (?P<scope>.+?)"
        r"(?=\b(?:truoc khi|sau khi|khi|de|theo)\b|$)"
    )
    _MANDATORY_SCOPE = re.compile(r"\bphai (?P<scope>.+)$")
    _LEGAL_REFERENCE = re.compile(
        r"(?i)\b(?P<kind>điều|khoản|điểm|chương|mục)\s+"
        r"(?P<value>\d+[a-zđ]*|[a-zđ]+|[ivxlcdm]+)\b"
    )
    _DOCUMENT_NUMBER = re.compile(r"\b\d{1,5}/[A-ZÀ-ỸĐ0-9-]{2,}\b")
    _DATE = re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])/(?:0?[1-9]|1[0-2])/\d{4}\b")
    _DOCUMENT_ID = re.compile(
        r"(?i)\b(?:document\s+id|mã\s+văn\s+bản)\s*:?\s*(?P<value>\d+)\b"
    )
    _REFERENCE_FIELDS = {
        "dieu": "article",
        "khoan": "clause",
        "diem": "point",
        "chuong": "chapter",
        "muc": "section",
    }

    def __init__(self, citation_validator: CitationValidator | None = None) -> None:
        self.citation_validator = citation_validator or CitationValidator()

    def validate(
        self,
        answer: str | GeneratedAnswer,
        source: RetrievalResult | LegalContext,
        *,
        evidences: list[LegalEvidence] | None = None,
        citation_result: CitationValidationResult | None = None,
    ) -> GroundingResult:
        text = answer.answer if isinstance(answer, GeneratedAnswer) else answer
        available = evidences if evidences is not None else source.evidences
        errors: list[str] = []
        if not text.strip():
            errors.append("Answer is empty.")
        if not available:
            errors.append("No supporting evidence is available.")
        citations = citation_result or self.citation_validator.validate(
            text, source, evidences=available
        )
        errors.extend(citations.errors)
        if text.strip() and available and citations.valid:
            cited = [
                available[citation_id - 1]
                for citation_id in citations.citation_ids
                if 1 <= citation_id <= len(available)
            ]
            if not cited and not self.citation_validator.is_safe_abstention(text):
                errors.append("Answer has no cited supporting evidence.")
            errors.extend(self._unsupported_metadata(text, cited))
            if not self.citation_validator.is_safe_abstention(text):
                errors.extend(self._unsupported_claims(text, available))
        return GroundingResult(grounded=not errors, errors=list(dict.fromkeys(errors)))

    def _unsupported_claims(
        self, answer: str, available: list[LegalEvidence]
    ) -> list[str]:
        """Check each deterministic claim against its explicitly scoped evidence."""
        errors: list[str] = []
        for claim, citation_ids in self._claim_units(answer):
            plain_claim = self._CITATION.sub("", claim).strip()
            if not self._content_tokens(plain_claim):
                continue
            if not citation_ids:
                errors.append(
                    "Unsupported claim detected without citation scope: "
                    f'"{plain_claim}"'
                )
                continue
            evidence = [
                available[citation_id - 1]
                for citation_id in citation_ids
                if 1 <= citation_id <= len(available)
            ]
            if evidence and not self._claim_supported(plain_claim, evidence):
                labels = ", ".join(f"[{value}]" for value in citation_ids)
                errors.append(
                    f'Unsupported claim detected in sentence: "{plain_claim}". '
                    f"Evidence {labels} does not support this detail."
                )
        return errors

    def _claim_units(self, answer: str) -> list[tuple[str, tuple[int, ...]]]:
        """Split prose/list claims and keep citation scope local and deterministic."""
        units: list[tuple[str, tuple[int, ...]]] = []
        list_scope: tuple[int, ...] = ()
        previous_was_blank = False
        for raw_line in answer.splitlines():
            line = raw_line.strip()
            if not line:
                previous_was_blank = True
                continue
            is_list_item = bool(self._LIST_ITEM.match(line))
            if previous_was_blank and not is_list_item:
                list_scope = ()
            previous_was_blank = False
            claims = [
                value.strip()
                for value in self._SENTENCE_BOUNDARY.split(line)
                if value.strip()
            ]
            for claim in claims:
                explicit = tuple(
                    dict.fromkeys(int(value) for value in self._CITATION.findall(claim))
                )
                scope = explicit or (list_scope if is_list_item else ())
                units.append((claim, scope))
                if explicit and claim.rstrip().endswith(":"):
                    list_scope = explicit
        return units

    def _claim_supported(
        self, claim: str, evidences: list[LegalEvidence]
    ) -> bool:
        claim_normalized = self._normalize_claim_text(claim)
        evidence_normalized = self._normalize_claim_text(
            "\n".join(self._evidence_support_text(item) for item in evidences)
        )

        # Numeric facts are high-information details. A citation cannot support a
        # number that is absent from its mapped evidence.
        claim_numbers = self._numbers(claim_normalized)
        evidence_numbers = self._numbers(evidence_normalized)
        if not claim_numbers.issubset(evidence_numbers):
            return False

        # ``tất cả các X`` may only emphasize an already-complete plural noun
        # phrase ``các X``. Keep this exception narrow: the exact scoped phrase
        # must occur in evidence, it must contain enough content to identify the
        # scope, and every occurrence in the claim must qualify. This preserves
        # rejection of new categories, actors, numbers, and conditions.
        if "tat ca" in claim_normalized:
            if not self._safe_all_emphasis(claim_normalized, evidence_normalized):
                return False

        # A permissive/rights statement cannot be promoted to an obligation.
        # Other obligation paraphrases are canonicalized by
        # ``_normalize_claim_text`` and retain their existing behavior.
        if self._promotes_permission_to_obligation(
            claim_normalized, evidence_normalized
        ):
            return False

        for pattern in self._DETAIL_PATTERNS:
            for detail in pattern.findall(claim_normalized):
                if detail not in evidence_normalized:
                    return False

        # Actor expansion such as "cả người đại diện cũ và mới" materially
        # changes who is obligated, even when the base actor overlaps.
        claim_tokens = self._content_tokens(claim_normalized)
        evidence_tokens = self._content_tokens(evidence_normalized)
        if {"ca", "cu", "moi"}.issubset(claim_tokens) and not {
            "ca",
            "cu",
            "moi",
        }.issubset(evidence_tokens):
            return False

        if not claim_tokens:
            return True
        supported = len(claim_tokens & evidence_tokens)
        coverage = supported / len(claim_tokens)
        # Short claims carry little redundancy, so every remaining content token
        # must be supported. Longer legal prose tolerates modest paraphrasing.
        threshold = 1.0 if len(claim_tokens) <= 3 else 0.72
        return coverage >= threshold

    @classmethod
    def _safe_all_emphasis(cls, claim: str, evidence: str) -> bool:
        matches = list(cls._ALL_EMPHASIS.finditer(claim))
        if not matches:
            return False
        for match in matches:
            scope = " ".join(match.group("scope").split())
            if len(cls._content_tokens(scope)) < 3:
                return False
            if f"cac {scope}" not in evidence:
                return False
        return True

    @classmethod
    def _promotes_permission_to_obligation(cls, claim: str, evidence: str) -> bool:
        for match in cls._MANDATORY_SCOPE.finditer(claim):
            scope = " ".join(match.group("scope").split())
            if f"co the {scope}" in evidence or f"co quyen {scope}" in evidence:
                return True
        return False

    @classmethod
    def _content_tokens(cls, value: str) -> set[str]:
        normalized = cls._normalize_claim_text(value)
        return {
            str(int(token)) if token.isdigit() else token
            for token in cls._TOKEN.findall(normalized)
            if token not in cls._STOP_WORDS and len(token) > 1
        }

    @staticmethod
    def _numbers(value: str) -> set[str]:
        numbers: set[str] = set()
        for token in re.findall(r"\b\d+(?:[.,]\d+)?\b", value):
            if token.isdigit():
                numbers.add(str(int(token)))
            else:
                numbers.add(token.replace(",", "."))
        return numbers

    @classmethod
    def _normalize_claim_text(cls, value: str) -> str:
        normalized = cls._normalize(cls._MARKDOWN.sub(" ", value))
        # Modality paraphrases do not change the underlying obligation.
        normalized = normalized.replace("co trach nhiem", "phai")
        normalized = normalized.replace("co nghia vu", "phai")
        return " ".join(normalized.split())

    @staticmethod
    def _evidence_support_text(evidence: LegalEvidence) -> str:
        return "\n".join(
            value
            for value in (
                evidence.document_name,
                evidence.chapter,
                evidence.section,
                evidence.article,
                evidence.clause,
                evidence.point,
                evidence.text,
            )
            if value
        )

    def _unsupported_metadata(
        self, answer: str, cited: list[LegalEvidence]
    ) -> list[str]:
        if not cited:
            return []
        references: dict[str, set[str]] = {
            field: set()
            for field in ("chapter", "section", "article", "clause", "point")
        }
        document_ids = {str(evidence.document_id) for evidence in cited}
        searchable_values: list[str] = []

        for evidence in cited:
            for field in references:
                value = getattr(evidence, field)
                if value:
                    references[field].add(self._canonical_reference(field, value))
            searchable_values.extend(
                value
                for value in (
                    evidence.document_name,
                    evidence.source_link,
                    evidence.chapter,
                    evidence.section,
                    evidence.article,
                    evidence.clause,
                    evidence.point,
                    evidence.text,
                )
                if value
            )

        # Legal text can explicitly quote another structured provision. Keep that
        # legitimate support, but extract it into the matching field's exact-value
        # set instead of searching one concatenated string.
        searchable_text = "\n".join(searchable_values)
        for match in self._LEGAL_REFERENCE.finditer(searchable_text):
            field = self._reference_field(match.group("kind"))
            references[field].add(
                self._canonical_reference(field, match.group(0))
            )

        document_numbers = {
            self._normalize(value)
            for value in self._DOCUMENT_NUMBER.findall(searchable_text)
        }
        dates = {
            self._normalize(value) for value in self._DATE.findall(searchable_text)
        }

        unsupported: list[str] = []
        for match in self._LEGAL_REFERENCE.finditer(answer):
            claim = match.group(0)
            field = self._reference_field(match.group("kind"))
            if self._canonical_reference(field, claim) not in references[field]:
                unsupported.append(f"Unsupported legal metadata in answer: {claim}")
        for claim in self._DOCUMENT_NUMBER.findall(answer):
            if self._normalize(claim) not in document_numbers:
                unsupported.append(f"Unsupported legal metadata in answer: {claim}")
        for claim in self._DATE.findall(answer):
            if self._normalize(claim) not in dates:
                unsupported.append(f"Unsupported legal metadata in answer: {claim}")
        for match in self._DOCUMENT_ID.finditer(answer):
            claim = match.group(0)
            if str(int(match.group("value"))) not in document_ids:
                unsupported.append(f"Unsupported legal metadata in answer: {claim}")
        return list(dict.fromkeys(unsupported))

    @classmethod
    def _reference_field(cls, kind: str) -> str:
        return cls._REFERENCE_FIELDS[cls._normalize(kind)]

    @classmethod
    def _canonical_reference(cls, field: str, value: str) -> str:
        """Return an exact, field-aware key for structured legal metadata.

        Numeric labels compare by numeric value (``Điều 05`` equals ``Điều 5``),
        while suffixes and letter labels remain exact (``Điểm a`` differs from
        ``Điểm aa``). The field is part of the key, so equal tokens in different
        legal levels cannot support one another.
        """
        normalized = cls._normalize(value)
        prefixes = {
            "chapter": "chuong",
            "section": "muc",
            "article": "dieu",
            "clause": "khoan",
            "point": "diem",
        }
        prefix = prefixes[field]
        token = normalized
        if token == prefix:
            token = ""
        elif token.startswith(f"{prefix} "):
            token = token[len(prefix) + 1 :].strip()

        numeric = re.fullmatch(r"0*(\d+)([a-z]*)", token)
        if numeric:
            token = f"{int(numeric.group(1))}{numeric.group(2)}"
        return f"{field}:{token}"

    @staticmethod
    def _normalize(value: str) -> str:
        decomposed = unicodedata.normalize("NFD", value.casefold())
        return " ".join(
            "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
            .replace("đ", "d")
            .split()
        )
