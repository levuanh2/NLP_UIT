"""Explicit composition root for the persisted local Legal RAG runtime."""

from pathlib import Path

from app.core.config import Settings
from app.generation.llm.qwen_generator import QwenGenerator
from app.generation.pipeline import GenerationPipeline
from app.generation.prompts.legal_answer import LegalPromptBuilder
from app.generation.validation.abstention_validator import AbstentionValidator
from app.generation.validation.citation_validator import CitationValidator
from app.generation.validation.grounding_validator import GroundingValidator
from app.indexing.embeddings.legal_embedding import VietnameseLegalEmbeddingModel
from app.indexing.lexical.bm25_index import BM25Index
from app.indexing.metadata_store.database import Database
from app.indexing.metadata_store.repository import LegalRepository
from app.indexing.vector_store.faiss_store import FAISSVectorStore
from app.retrieval.context.context_builder import LegalContextBuilder
from app.retrieval.context.parent_expander import ParentContextExpander
from app.retrieval.dense.retriever import DenseRetriever
from app.retrieval.filters.metadata_filter import MetadataFilter
from app.retrieval.fusion.rrf import ReciprocalRankFusion
from app.retrieval.lexical.retriever import LexicalRetriever
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query.analyzer import QueryAnalyzer
from app.retrieval.query.metadata_extractor import QueryMetadataExtractor
from app.retrieval.reranking.identity import IdentityReranker
from app.retrieval.reranking.vietnamese_reranker import VietnameseReranker
from app.services.legal_rag_service import LegalRAGService


def build_local_rag_service(
    settings: Settings,
    faiss_path: Path,
    bm25_path: Path,
    metadata_path: Path,
    embedding_model_path: Path,
    reranker_model_path: Path,
    llm_model_path: Path,
) -> LegalRAGService:
    for path, label in (
        (faiss_path, "FAISS index"),
        (bm25_path, "BM25 index"),
        (metadata_path, "metadata database"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    for path, label in (
        (embedding_model_path, "embedding model"),
        (llm_model_path, "LLM model"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    database = Database(metadata_path)
    database.initialize()
    repository = LegalRepository(database)
    embedding = VietnameseLegalEmbeddingModel(
        str(embedding_model_path),
        settings.embedding_device,
        True,
        "query: ",
        "passage: ",
    )
    vector_store = FAISSVectorStore()
    vector_store.load(faiss_path)
    bm25 = BM25Index()
    bm25.load(bm25_path)
    reranker = (
        VietnameseReranker(
            str(reranker_model_path),
            settings.reranker_device,
            True,
            settings.model_trust_remote_code,
        )
        if settings.rerank_enabled
        else IdentityReranker()
    )
    if settings.rerank_enabled and not reranker_model_path.is_dir():
        raise FileNotFoundError(f"Reranker model does not exist: {reranker_model_path}")
    retrieval = RetrievalPipeline(
        query_analyzer=QueryAnalyzer(QueryMetadataExtractor()),
        metadata_filter=MetadataFilter(
            repository,
            settings.metadata_filter_enabled,
            settings.metadata_filter_min_confidence,
            settings.metadata_filter_fallback_to_full_corpus,
        ),
        dense_retriever=DenseRetriever(embedding, vector_store, repository),
        lexical_retriever=LexicalRetriever(bm25, repository),
        fusion=ReciprocalRankFusion(settings.rrf_k),
        reranker=reranker,
        parent_expander=ParentContextExpander(repository),
        context_builder=LegalContextBuilder(),
        dense_top_n=settings.dense_top_n,
        lexical_top_n=settings.bm25_top_n,
        fusion_top_n=settings.fusion_top_n,
        rerank_top_k=settings.rerank_top_k,
    )
    generation = GenerationPipeline(
        generator=QwenGenerator(
            str(llm_model_path),
            settings.model_device,
            settings.model_dtype,
            True,
            settings.model_trust_remote_code,
            settings.max_new_tokens,
            settings.temperature,
            settings.top_p,
            settings.do_sample,
        ),
        prompt_builder=LegalPromptBuilder(),
        citation_validator=CitationValidator(),
        grounding_validator=GroundingValidator(),
        abstention_validator=AbstentionValidator(
            "Không tìm thấy đủ căn cứ pháp lý trong các văn bản được cung cấp."
        ),
        require_citation=settings.require_citation,
        grounded_only=settings.grounded_only,
    )
    return LegalRAGService(retrieval, generation)
