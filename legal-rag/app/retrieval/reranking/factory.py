"""Reranker factory."""

from app.retrieval.reranking.base import BaseReranker
from app.retrieval.reranking.vietnamese_reranker import VietnameseReranker


class RerankerFactory:
    @staticmethod
    def create(
        provider: str,
        model_name: str,
        device: str,
        local_files_only: bool,
        trust_remote_code: bool,
        repository=None,
        parameter_budget_approved: bool = False,
    ) -> BaseReranker:
        if provider != "local_transformers":
            raise ValueError(f"Unsupported reranker provider: {provider}")
        return VietnameseReranker(
            model_name,
            device,
            local_files_only,
            trust_remote_code,
            repository=repository,
            parameter_budget_approved=parameter_budget_approved,
        )
