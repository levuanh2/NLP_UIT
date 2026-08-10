"""Production composition for the approved all-local Legal RAG models."""

import time
from dataclasses import dataclass

from app.core.config import Settings
from app.generation.citation_repair import CitationRepair
from app.generation.llm.base import BaseLLMGenerator
from app.generation.llm.factory import LLMGeneratorFactory
from app.generation.pipeline import GenerationPipeline
from app.generation.prompts.citation_repair import CitationRepairPromptBuilder
from app.generation.prompts.legal_answer import LegalPromptBuilder
from app.generation.validation.abstention_validator import AbstentionValidator
from app.generation.validation.citation_validator import CitationValidator
from app.generation.validation.grounding_validator import GroundingValidator
from app.indexing.embeddings.base import BaseEmbeddingModel
from app.indexing.embeddings.factory import EmbeddingModelFactory
from app.indexing.metadata_store.database import Database
from app.indexing.metadata_store.repository import LegalRepository
from app.retrieval.active_index import ActiveIndex
from app.retrieval.context.context_builder import LegalContextBuilder
from app.retrieval.context.parent_expander import ParentContextExpander
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.fusion.rrf import RRFFusion
from app.retrieval.lexical.retriever import BM25Retriever
from app.retrieval.metadata_filter import MetadataFilter
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.reranking.base import BaseReranker
from app.retrieval.reranking.factory import RerankerFactory
from app.services.legal_rag_service import LegalRAGService

APPROVED_MODELS = {
    "embedding": "bqbbao6/vietnamese-legal-embedding",
    "reranker": "AITeamVN/Vietnamese_Reranker",
    "llm": "AITeamVN/Vi-Qwen2-3B-RAG",
}
PARAMETER_BUDGET_LIMIT = 4_000_000_000


@dataclass
class LocalRAGRuntime:
    service: LegalRAGService
    embedding: BaseEmbeddingModel
    reranker: BaseReranker
    llm: BaseLLMGenerator
    parameter_counts: dict[str, int]
    load_seconds: dict[str, float]
    device: str
    dtype: str

    def close(self) -> None:
        self.llm.unload()
        self.reranker.unload()
        self.embedding.unload()


def build_local_rag_runtime(
    settings: Settings, *, trace: bool = False
) -> LocalRAGRuntime:
    configured = {
        "embedding": settings.embedding_model_name,
        "reranker": settings.reranker_model_name,
        "llm": settings.llm_model_name,
    }
    if configured != APPROVED_MODELS:
        raise RuntimeError(
            f"Configured models differ from BTC-approved models: {configured}"
        )
    active = ActiveIndex(settings.index_root_dir)
    database = Database(active.sqlite_path)
    database.initialize()
    repository = LegalRepository(database)
    embedding = EmbeddingModelFactory.create(
        provider="sentence_transformers",
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        local_files_only=settings.model_local_files_only,
    )
    reranker: BaseReranker | None = None
    llm: BaseLLMGenerator | None = None
    try:
        load_seconds: dict[str, float] = {}
        started = time.perf_counter()
        embedding.load()
        load_seconds["embedding"] = time.perf_counter() - started
        reranker = RerankerFactory.create(
            provider="local_transformers",
            model_name=settings.reranker_model_name,
            device=settings.reranker_device,
            local_files_only=settings.model_local_files_only,
            trust_remote_code=settings.model_trust_remote_code,
            repository=repository,
            parameter_budget_approved=False,
        )
        started = time.perf_counter()
        reranker.load()
        load_seconds["reranker"] = time.perf_counter() - started
        llm = LLMGeneratorFactory.create(
            provider="local_transformers",
            model_name=settings.llm_model_name,
            device=settings.model_device,
            dtype=settings.model_dtype,
            local_files_only=settings.model_local_files_only,
            trust_remote_code=settings.model_trust_remote_code,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            do_sample=settings.do_sample,
            min_new_tokens=settings.min_new_tokens,
            repetition_penalty=settings.repetition_penalty,
        )
        started = time.perf_counter()
        llm.load()
        load_seconds["llm"] = time.perf_counter() - started
        parameter_counts = {
            "embedding": _parameter_count(embedding),
            "reranker": _parameter_count(reranker),
            "llm": _parameter_count(llm),
        }
        total_parameters = sum(parameter_counts.values())
        parameter_counts["total"] = total_parameters
        verify_parameter_budget(parameter_counts)
        verify_budget = getattr(reranker, "verify_parameter_budget", None)
        if callable(verify_budget):
            verify_budget(total_parameters)
        retrieval = RetrievalPipeline(
            QueryAnalyzer(),
            MetadataFilter(
                repository,
                enabled=settings.metadata_filter_enabled,
                min_confidence=settings.metadata_filter_min_confidence,
            ),
            DenseRetriever(
                embedding, repository=repository, index_root=settings.index_root_dir
            ),
            BM25Retriever(repository=repository, index_root=settings.index_root_dir),
            RRFFusion(),
            reranker,
            ParentContextExpander(
                repository,
                neighbor_window=settings.context_neighbor_window,
                max_parents_per_document=settings.max_parents_per_document,
            ),
            LegalContextBuilder(max_tokens=settings.context_max_tokens),
            dense_top_k=settings.dense_top_n,
            bm25_top_k=settings.bm25_top_n,
            rrf_k=settings.rrf_k,
            rrf_top_k=settings.fusion_top_n,
            reranker_enabled=settings.rerank_enabled,
            reranker_top_k=settings.rerank_top_k,
            trace=trace or settings.retrieval_trace,
        )
        citation_validator = CitationValidator(
            require_citation=settings.require_citation
        )
        generation = GenerationPipeline(
            llm,
            LegalPromptBuilder(
                max_context_tokens=settings.llm_max_context_tokens,
                reserved_generation_tokens=settings.max_new_tokens,
            ),
            citation_validator,
            GroundingValidator(citation_validator),
            AbstentionValidator(),
            CitationRepair(
                llm,
                CitationRepairPromptBuilder(),
                enabled=settings.llm_citation_repair_enabled,
                max_retries=settings.llm_citation_repair_max_retries,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                max_context_tokens=settings.llm_max_context_tokens,
            ),
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            trace=trace or settings.generation_trace,
        )
        return LocalRAGRuntime(
            service=LegalRAGService(retrieval, generation),
            embedding=embedding,
            reranker=reranker,
            llm=llm,
            parameter_counts=parameter_counts,
            load_seconds=load_seconds,
            device=str(getattr(llm, "_resolved_device", settings.model_device)),
            dtype=_model_dtype(llm),
        )
    except Exception:
        if llm is not None:
            llm.unload()
        if reranker is not None:
            reranker.unload()
        embedding.unload()
        raise


def _parameter_count(model_adapter: object) -> int:
    model = getattr(model_adapter, "_model", None)
    if model is None:
        raise RuntimeError(
            f"Cannot verify parameters for unloaded {type(model_adapter).__name__}"
        )
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise RuntimeError(
            f"Configured model {type(model_adapter).__name__} has no parameters()"
        )
    return sum(parameter.numel() for parameter in parameters())


def verify_parameter_budget(parameter_counts: dict[str, int]) -> None:
    """Fail before inference unless all loaded local models total under 4B."""
    required = ("embedding", "reranker", "llm")
    missing = [name for name in required if name not in parameter_counts]
    if missing:
        raise RuntimeError(
            "Cannot verify competition parameter budget; missing counts: "
            + ", ".join(missing)
        )
    counts = {name: parameter_counts[name] for name in required}
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"Model parameter counts must be positive: {counts}")
    total = sum(counts.values())
    if total >= PARAMETER_BUDGET_LIMIT:
        raise RuntimeError(
            "Configured embedding + reranker + LLM models exceed the strict "
            f"<4B competition ceiling: {total:,} >= {PARAMETER_BUDGET_LIMIT:,}"
        )


def _model_dtype(model_adapter: object) -> str:
    model = getattr(model_adapter, "_model", None)
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "unknown"
    first = next(parameters(), None)
    return str(first.dtype) if first is not None else "unknown"
