"""Dense retriever skeleton."""

from app.domain.retrieval import RetrievalCandidate
from app.indexing.embeddings.base import BaseEmbeddingModel
from app.indexing.metadata_store.repository import LegalRepository
from app.indexing.vector_store.base import BaseVectorStore


class DenseRetriever:
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        repository: LegalRepository,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.repository = repository

    def retrieve(
        self,
        query: str,
        top_n: int,
        allowed_ids: set[str] | None = None,
    ) -> list[RetrievalCandidate]:
        """Embed query, search FAISS, and load chunk metadata."""
        # TODO(phase-implementation):
        # Implement dense retrieval and score-to-domain mapping.
        raise NotImplementedError
