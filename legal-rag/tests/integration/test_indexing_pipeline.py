"""Indexing orchestration tests."""

import numpy as np

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.metadata import LegalMetadata
from app.indexing.pipeline import IndexingPipeline


class _Embedding:
    def load(self) -> None:
        pass

    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class _Vector:
    def create(self, dimension: int) -> None:
        self.dimension = dimension

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        self.ids = ids


class _BM25:
    def build(self, texts: list[str], ids: list[str]) -> None:
        self.ids = ids


class _Repository:
    def save_chunks(
        self, parents: list[ParentChunk], children: list[ChildChunk]
    ) -> None:
        self.children = children


def test_indexing_pipeline_persists_all_child_chunks() -> None:
    metadata = LegalMetadata(
        document_id=1,
        document_name="doc",
        source_link="",
        chapter=None,
        section=None,
        article="1",
        clause=None,
        point=None,
    )
    parent = ParentChunk(
        parent_id="p",
        document_id=1,
        chapter=None,
        section=None,
        article="1",
        text="parent",
        token_count=1,
    )
    child = ChildChunk(
        child_id="c",
        parent_id="p",
        document_id=1,
        chapter=None,
        section=None,
        article="1",
        clause=None,
        point=None,
        original_text="child",
        embedding_text="child",
        token_count=1,
        metadata=metadata,
    )
    vector, bm25, repository = _Vector(), _BM25(), _Repository()
    pipeline = IndexingPipeline(_Embedding(), vector, bm25, repository)  # type: ignore[arg-type]
    result = pipeline.build([child], [parent])
    assert result.child_count == 1 and result.embedding_dimension == 2
    assert vector.ids == bm25.ids == ["c"]
    assert repository.children == [child]
