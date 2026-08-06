"""Embedding model factory."""

from app.indexing.embeddings.base import BaseEmbeddingModel
from app.indexing.embeddings.legal_embedding import VietnameseLegalEmbeddingModel


class EmbeddingModelFactory:
    @staticmethod
    def create(
        provider: str,
        model_name: str,
        device: str,
        local_files_only: bool,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> BaseEmbeddingModel:
        if provider != "sentence_transformers":
            raise ValueError(f"Unsupported embedding provider: {provider}")
        return VietnameseLegalEmbeddingModel(
            model_name=model_name,
            device=device,
            local_files_only=local_files_only,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
