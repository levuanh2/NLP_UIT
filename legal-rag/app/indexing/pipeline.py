"""Indexing pipeline composition root."""

from pydantic import BaseModel

from app.domain.chunks import ChildChunk, ParentChunk
from app.indexing.embeddings.base import BaseEmbeddingModel
from app.indexing.lexical.bm25_index import BM25Index
from app.indexing.metadata_store.repository import LegalRepository
from app.indexing.vector_store.base import BaseVectorStore


class IndexingResult(BaseModel):
    child_count: int
    parent_count: int
    embedding_dimension: int | None = None


class IndexingPipeline:
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        bm25_index: BM25Index,
        metadata_repository: LegalRepository,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.metadata_repository = metadata_repository

    def build(
        self,
        child_chunks: list[ChildChunk],
        parent_chunks: list[ParentChunk],
    ) -> IndexingResult:
        """Embed children, build vector/BM25 indexes, and persist metadata."""
        # TODO(phase-implementation):
        # Implement indexing orchestration and atomic artifact publishing.
        raise NotImplementedError
