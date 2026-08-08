"""Parent context expansion tests."""

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.metadata import LegalMetadata
from app.domain.retrieval import RetrievalCandidate
from app.retrieval.context.parent_expander import ParentContextExpander


def _metadata() -> LegalMetadata:
    return LegalMetadata(
        document_id=1,
        document_name="Luật mẫu",
        source_link="https://example.test",
        chapter=None,
        section=None,
        article="1",
        clause="1",
        point=None,
    )


class _Repository:
    def get_child(self, child_id: str) -> ChildChunk:
        return ChildChunk(
            child_id=child_id,
            parent_id="parent",
            document_id=1,
            chapter=None,
            section=None,
            article="1",
            clause="1",
            point=None,
            original_text="child",
            embedding_text="child",
            token_count=1,
            metadata=_metadata(),
        )

    def get_parent(self, parent_id: str) -> ParentChunk:
        return ParentChunk(
            parent_id=parent_id,
            document_id=1,
            chapter=None,
            section=None,
            article="1",
            text="parent text",
            token_count=2,
        )


def test_parent_expander_deduplicates_parent() -> None:
    candidates = [
        RetrievalCandidate(child_id=value, text=value, metadata=_metadata())
        for value in ("a", "b")
    ]
    result = ParentContextExpander(_Repository()).expand(candidates)  # type: ignore[arg-type]
    assert len(result) == 1
    assert result[0].text == "parent text"
