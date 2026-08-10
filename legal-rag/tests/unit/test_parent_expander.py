"""Parent and same-parent neighbor expansion tests."""

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.metadata import LegalMetadata
from app.domain.retrieval import RetrievalCandidate
from app.retrieval.context.parent_expander import ParentContextExpander


class FakeRepository:
    def __init__(self, parents: list[ParentChunk], children: list[ChildChunk]) -> None:
        self.parents = {parent.parent_id: parent for parent in parents}
        self.children = {child.child_id: child for child in children}

    def get_child(self, child_id: str) -> ChildChunk | None:
        return self.children.get(child_id)

    def get_parent(self, parent_id: str) -> ParentChunk | None:
        return self.parents.get(parent_id)

    def get_neighbor_children(self, child_id: str, window: int) -> list[ChildChunk]:
        child = self.children[child_id]
        return sorted(
            [
                item
                for item in self.children.values()
                if item.parent_id == child.parent_id
                and abs(item.position - child.position) <= window
            ],
            key=lambda item: item.position,
        )


def make_parent(parent_id: str, document_id: int, position: int) -> ParentChunk:
    return ParentChunk(
        parent_id=parent_id,
        document_id=document_id,
        chapter=None,
        section=None,
        article=f"Điều {position + 1}",
        position=position,
        text="parent",
        token_count=1,
    )


def make_child(parent: ParentChunk, position: int, text: str) -> ChildChunk:
    metadata = LegalMetadata(
        document_id=parent.document_id,
        document_name=f"doc-{parent.document_id}",
        source_link=None,
        chapter=None,
        section=None,
        article=parent.article,
        clause=None,
        point=None,
    )
    return ChildChunk(
        child_id=f"{parent.parent_id}:child:{position}",
        parent_id=parent.parent_id,
        document_id=parent.document_id,
        document_name=metadata.document_name,
        chapter=None,
        section=None,
        article=parent.article,
        clause=None,
        point=None,
        position=position,
        text=text,
        embedding_text=text,
        token_count=len(text.split()),
        metadata=metadata,
    )


def candidate(child: ChildChunk) -> RetrievalCandidate:
    assert child.metadata is not None
    return RetrievalCandidate(
        child_id=child.child_id,
        text=child.text,
        metadata=child.metadata,
    )


def test_parent_neighbor_expansion_is_same_parent_and_deduplicated() -> None:
    parent = make_parent("parent-1", 1, 0)
    other = make_parent("parent-2", 2, 0)
    children = [
        make_child(parent, 0, "một hai ba"),
        make_child(parent, 1, "ba bốn năm"),
        make_child(other, 2, "không được lấy"),
    ]
    repository = FakeRepository([parent, other], children)
    expander = ParentContextExpander(repository, neighbor_window=1)  # type: ignore[arg-type]

    evidences = expander.expand([candidate(children[1]), candidate(children[0])])

    assert len(evidences) == 1
    assert evidences[0].parent_id == parent.parent_id
    assert "không được lấy" not in evidences[0].text
    assert evidences[0].text.count("ba") == 1


def test_max_parents_per_document_preserves_diversity() -> None:
    parents = [make_parent(f"p-{index}", 1, index) for index in range(3)]
    other = make_parent("other", 2, 0)
    children = [make_child(parent, 0, parent.parent_id) for parent in parents + [other]]
    repository = FakeRepository(parents + [other], children)
    expander = ParentContextExpander(repository, max_parents_per_document=2)  # type: ignore[arg-type]

    evidences = expander.expand([candidate(child) for child in children])

    assert [evidence.document_id for evidence in evidences] == [1, 1, 2]


def test_all_direct_children_from_same_parent_survive_neighbor_union() -> None:
    parent = make_parent("parent-direct", 1, 0)
    children = [
        make_child(parent, position, f"C{position + 1}") for position in range(3)
    ]
    repository = FakeRepository([parent], children)
    expander = ParentContextExpander(repository, neighbor_window=1)  # type: ignore[arg-type]

    evidences = expander.expand([candidate(child) for child in children])

    assert len(evidences) == 1
    assert evidences[0].text.split() == ["C1", "C2", "C3"]


def test_overlapping_neighbor_windows_are_unioned_at_child_level() -> None:
    parent = make_parent("parent-union", 1, 0)
    children = [
        make_child(parent, position, f"C{position + 1}") for position in range(5)
    ]
    repository = FakeRepository([parent], children)
    expander = ParentContextExpander(repository, neighbor_window=1)  # type: ignore[arg-type]

    evidences = expander.expand([candidate(children[1]), candidate(children[3])])

    assert len(evidences) == 1
    assert evidences[0].text.split() == ["C1", "C2", "C3", "C4", "C5"]


def test_neighbor_expansion_never_crosses_parent_boundary() -> None:
    first_parent = make_parent("parent-one", 1, 0)
    second_parent = make_parent("parent-two", 1, 1)
    first_children = [
        make_child(first_parent, 0, "C1"),
        make_child(first_parent, 1, "C2"),
    ]
    second_children = [
        make_child(second_parent, 2, "C3"),
        make_child(second_parent, 3, "C4"),
    ]
    repository = FakeRepository(
        [first_parent, second_parent], first_children + second_children
    )
    expander = ParentContextExpander(repository, neighbor_window=1)  # type: ignore[arg-type]

    evidences = expander.expand([candidate(first_children[1])])

    assert len(evidences) == 1
    assert evidences[0].text.split() == ["C1", "C2"]
    assert "C3" not in evidences[0].text
