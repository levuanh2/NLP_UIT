"""Hierarchical and fallback chunking tests."""

from app.domain.documents import LegalDocument
from app.ingestion.chunking.parent_child import ParentChildChunker
from app.ingestion.cleaners.legal_text_cleaner import LegalTextCleaner
from app.ingestion.structure.extractor import LegalStructureExtractor


def make_document(text: str) -> LegalDocument:
    document = LegalDocument(
        document_id=10,
        document_name="Luật thử nghiệm",
        source_link="https://example.test/10",
        raw_text=text,
        cleaned_text=LegalTextCleaner().clean(text),
    )
    return LegalStructureExtractor().extract(document)


def test_article_parent_and_clause_point_children() -> None:
    document = make_document(
        "Chương I\nĐiều 3. Nội dung\n1. Khoản một\na) Điểm a\nb) Điểm b\n2. Khoản hai"
    )

    parents, children = ParentChildChunker().chunk(document)

    assert len(parents) == 1
    assert parents[0].article == "Điều 3"
    assert [child.point for child in children[:2]] == ["Điểm a", "Điểm b"]
    assert children[-1].clause == "Khoản 2"
    assert all(child.parent_id == parents[0].parent_id for child in children)


def test_long_article_parent_split() -> None:
    document = make_document("Điều 1. Dài\n" + " ".join(["từ"] * 90))
    chunker = ParentChildChunker(
        parent_target_tokens=20,
        parent_max_tokens=30,
        child_target_tokens=8,
        child_max_tokens=12,
        child_overlap_tokens=2,
    )

    parents, _ = chunker.chunk(document)

    assert len(parents) > 1
    assert all(parent.chunking_method == "article_segment" for parent in parents)


def test_token_window_fallback() -> None:
    document = make_document(" ".join(["không-cấu-trúc"] * 40))
    chunker = ParentChildChunker(
        parent_target_tokens=20,
        parent_max_tokens=30,
        child_target_tokens=8,
        child_max_tokens=12,
        child_overlap_tokens=2,
    )

    parents, children = chunker.chunk(document)

    assert len(parents) == 2
    assert all(child.chunking_method == "token_window" for child in children)


def test_child_embedding_path_position_and_links() -> None:
    document = make_document("Điều 7. Nội dung\n1. Một\n2. Hai\n3. Ba")

    _, children = ParentChildChunker().chunk(document)

    assert "Luật thử nghiệm > Điều 7 > Khoản 1" in children[0].embedding_text
    assert [child.position for child in children] == [0, 1, 2]
    assert children[0].previous_child_id is None
    assert children[0].next_child_id == children[1].child_id
    assert children[1].previous_child_id == children[0].child_id
    assert children[-1].next_child_id is None


def test_large_fallback_document_yields_bounded_groups() -> None:
    document = make_document(" ".join(["nội-dung"] * 10_000))
    chunker = ParentChildChunker(
        parent_target_tokens=100,
        parent_max_tokens=120,
        child_target_tokens=25,
        child_max_tokens=32,
        child_overlap_tokens=5,
    )

    group_count = 0
    for parent, children in chunker.iter_chunk_groups(document):
        group_count += 1
        assert parent.token_count is not None and parent.token_count <= 100
        assert len(children) <= 5

    assert group_count == 100
