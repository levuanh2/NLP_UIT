"""Citation metric skeletons."""

from app.domain.generation import Citation


def citation_precision(
    predicted: list[Citation], supported_evidence_ids: set[str]
) -> float:
    # TODO(phase-implementation):
    # Compute supported-citation precision with an explicit empty policy.
    raise NotImplementedError
