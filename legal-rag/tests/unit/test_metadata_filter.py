"""Confidence-gated metadata filtering tests."""

from app.domain.queries import QueryMetadata
from app.retrieval.filters.metadata_filter import MetadataFilter


class _Repository:
    def __init__(self, values: set[str]) -> None:
        self.values = values

    def filter_child_ids(self, metadata: QueryMetadata) -> set[str]:
        del metadata
        return self.values


def test_metadata_filter_applies_when_confident() -> None:
    value = QueryMetadata(article="7", confidence=0.9)
    filtering = MetadataFilter(_Repository({"a"}), True, 0.8, True)  # type: ignore[arg-type]
    assert filtering.allowed_ids(value) == {"a"}


def test_metadata_filter_skips_when_low_confidence() -> None:
    value = QueryMetadata(article="7", confidence=0.5)
    filtering = MetadataFilter(_Repository({"a"}), True, 0.8, True)  # type: ignore[arg-type]
    assert filtering.allowed_ids(value) is None


def test_metadata_filter_falls_back_to_full_corpus() -> None:
    value = QueryMetadata(article="7", confidence=0.9)
    filtering = MetadataFilter(_Repository(set()), True, 0.8, True)  # type: ignore[arg-type]
    assert filtering.allowed_ids(value) is None
