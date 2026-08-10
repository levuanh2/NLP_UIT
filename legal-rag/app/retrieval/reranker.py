"""Public local reranker API."""

from app.retrieval.reranking.base import BaseReranker
from app.retrieval.reranking.vietnamese_reranker import VietnameseReranker

Reranker = VietnameseReranker

__all__ = ["BaseReranker", "Reranker", "VietnameseReranker"]
