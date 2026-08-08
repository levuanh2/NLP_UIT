"""Vietnamese legal query normalization and analysis."""

import re
import unicodedata

from app.domain.queries import QueryAnalysis
from app.retrieval.query.metadata_extractor import QueryMetadataExtractor


class QueryAnalyzer:
    def __init__(self, metadata_extractor: QueryMetadataExtractor) -> None:
        self.metadata_extractor = metadata_extractor

    def analyze(self, query: str) -> QueryAnalysis:
        normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFC", query)).strip()
        if not normalized:
            raise ValueError("Legal query must not be empty.")
        metadata = self.metadata_extractor.extract(normalized)
        lowered = normalized.casefold()
        if any(term in lowered for term in ("xử phạt", "phạt", "mức phạt")):
            intent = "sanction"
        elif any(term in lowered for term in ("thủ tục", "hồ sơ", "mẫu")):
            intent = "procedure"
        elif any(term in lowered for term in ("điều kiện", "yêu cầu")):
            intent = "condition"
        else:
            intent = "legal_information"
        useful = any(
            (
                metadata.document_name,
                metadata.document_number,
                metadata.document_type,
                metadata.issued_year,
                metadata.article,
                metadata.clause,
            )
        )
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            metadata=metadata,
            should_apply_metadata_filter=useful and metadata.confidence > 0.0,
        )
