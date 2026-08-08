"""Parent-child chunking tests."""

from app.domain.documents import LegalDocument
from app.ingestion.chunking.parent_child import ParentChildChunker
from app.ingestion.structure.extractor import LegalStructureExtractor


def test_parent_child_chunker_creates_stable_linked_chunks() -> None:
    document = LegalDocument(
        document_id=7,
        document_name="Nghị định mẫu",
        source_link="https://example.test/7",
        raw_text="ĐIỀU 7. Xử phạt\n1. Hành vi vi phạm bị phạt tiền.",
        cleaned_text="ĐIỀU 7. Xử phạt\n1. Hành vi vi phạm bị phạt tiền.",
    )
    document = LegalStructureExtractor().extract(document)
    chunker = ParentChildChunker(100, 120, 40, 60)

    parents, children = chunker.chunk(document)
    repeated = chunker.chunk(document)

    assert parents and children
    assert {item.parent_id for item in children} <= {item.parent_id for item in parents}
    assert [item.parent_id for item in parents] == [
        item.parent_id for item in repeated[0]
    ]
    assert children[-1].metadata.document_name == "Nghị định mẫu"
    assert children[0].article == "7"
    assert children[0].clause == "1"
