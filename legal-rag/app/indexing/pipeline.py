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
        child_ids = [item.child_id for item in child_chunks]
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("Child chunk IDs must be unique before indexing.")
        texts = [item.embedding_text or item.original_text for item in child_chunks]
        self.embedding_model.load()
        dimension = self.embedding_model.dimension()
        vectors = self.embedding_model.embed_documents(texts)
        if vectors.shape != (len(child_chunks), dimension):
            raise ValueError("Embedding output shape does not match child chunks.")
        self.vector_store.create(dimension)
        self.vector_store.add(vectors, child_ids)
        self.bm25_index.build([item.original_text for item in child_chunks], child_ids)
        self.metadata_repository.save_chunks(parent_chunks, child_chunks)
        return IndexingResult(
            child_count=len(child_chunks),
            parent_count=len(parent_chunks),
            embedding_dimension=dimension,
        )
