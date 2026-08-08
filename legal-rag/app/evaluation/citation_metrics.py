"""Citation metrics."""

from app.domain.generation import Citation


def citation_precision(
    predicted: list[Citation], supported_evidence_ids: set[str]
) -> float:
    identifiers = [item.evidence_id for item in predicted if item.evidence_id]
    if not identifiers:
        return 0.0
    return sum(item in supported_evidence_ids for item in identifiers) / len(
        identifiers
    )
