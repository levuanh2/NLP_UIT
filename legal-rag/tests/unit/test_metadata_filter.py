"""Optional metadata-filter behavior tests."""

from app.domain.queries import QueryMetadata
from app.retrieval.metadata_filter import MetadataFilter


class FakeRepository:
    def __init__(self, ids: set[str]) -> None:
        self.ids = ids
        self.received: QueryMetadata | None = None

    def filter_child_ids(self, metadata: QueryMetadata) -> set[str]:
        self.received = metadata
        return self.ids


def test_metadata_filter_document() -> None:
    repository = FakeRepository({"child-1"})
    metadata_filter = MetadataFilter(repository)  # type: ignore[arg-type]
    metadata = QueryMetadata(raw_query="q", document_id=123, confidence=1.0)

    result = metadata_filter.build_filter(metadata)

    assert result.candidate_ids == {"child-1"}
    assert result.authoritative is True
    assert repository.received is metadata


def test_metadata_filter_article() -> None:
    repository = FakeRepository({"child-37"})
    metadata_filter = MetadataFilter(repository)  # type: ignore[arg-type]

    result = metadata_filter.build_filter(
        QueryMetadata(raw_query="q", article="Điều 37", confidence=1.0)
    )

    assert result.applied is True
    assert result.fields == ("article",)
    assert result.candidate_ids == {"child-37"}


def test_metadata_filter_optional() -> None:
    repository = FakeRepository({"must-not-be-used"})
    metadata_filter = MetadataFilter(repository)  # type: ignore[arg-type]

    result = metadata_filter.build_filter(
        QueryMetadata(raw_query="Điều kiện để doanh nghiệp được hoạt động")
    )

    assert result.applied is False
    assert result.candidate_ids is None
    assert repository.received is None


def test_metadata_filter_low_confidence_is_not_applied() -> None:
    repository = FakeRepository({"must-not-be-used"})
    metadata_filter = MetadataFilter(repository, min_confidence=0.8)  # type: ignore[arg-type]

    result = metadata_filter.build_filter(
        QueryMetadata(raw_query="q", article="Điều 1", confidence=0.5)
    )

    assert result.applied is False
    assert repository.received is None
