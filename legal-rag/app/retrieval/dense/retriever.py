"""Dense retrieval over every FAISS shard in the active index."""

from pathlib import Path

from app.domain.retrieval import RetrievalCandidate
from app.indexing.embeddings.base import BaseEmbeddingModel
from app.indexing.metadata_store.repository import LegalRepository
from app.indexing.vector_store.base import BaseVectorStore
from app.indexing.vector_store.faiss_store import FAISSVectorStore
from app.retrieval.active_index import ActiveIndex


class DenseRetriever:
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore | None = None,
        repository: LegalRepository | None = None,
        *,
        index_root: Path | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.repository = repository
        self.active_index = ActiveIndex(index_root) if index_root is not None else None
        self._stores: list[BaseVectorStore] = []
        if vector_store is not None:
            self._stores.append(vector_store)
        if self.active_index is not None:
            if self.active_index.manifest.embedding_model != getattr(
                embedding_model,
                "model_name",
                self.active_index.manifest.embedding_model,
            ):
                raise RuntimeError(
                    "Configured embedding model differs from active index manifest"
                )
            for path in self.active_index.faiss_shards:
                store = FAISSVectorStore()
                store.load(path)
                index_dimension = int(store._require_index().d)
                if index_dimension != self.active_index.manifest.embedding_dimension:
                    raise RuntimeError(
                        "FAISS dimension differs from active index manifest: "
                        f"{index_dimension} != "
                        f"{self.active_index.manifest.embedding_dimension}"
                    )
                self._stores.append(store)
        if not self._stores:
            raise ValueError("DenseRetriever requires an active index or vector store")

    @property
    def active_index_version(self) -> str | None:
        return self.active_index.version if self.active_index else None

    def retrieve(
        self,
        query: str,
        *,
        candidate_ids: set[str] | None = None,
        top_k: int = 20,
    ) -> list[RetrievalCandidate]:
        if top_k <= 0:
            return []
        vector = self.embedding_model.embed_query(query)
        actual_dimension = int(vector.reshape(-1).shape[0])
        if self.active_index is not None:
            expected = self.active_index.manifest.embedding_dimension
            if actual_dimension != expected:
                raise RuntimeError(
                    "Embedding dimension mismatch: "
                    f"query={actual_dimension}, index={expected}"
                )
        hits: dict[str, float] = {}
        for store in self._stores:
            for child_id, score in store.search(vector, top_k, candidate_ids):
                hits[child_id] = max(score, hits.get(child_id, float("-inf")))
        ordered = sorted(hits.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            RetrievalCandidate(
                child_id=child_id,
                score=score,
                source="dense",
                rank=rank,
                dense_score=score,
            )
            for rank, (child_id, score) in enumerate(ordered, start=1)
        ]
