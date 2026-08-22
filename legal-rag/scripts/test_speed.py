import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Loading RAG configurations...")
    from app.core.config import get_settings
    settings = get_settings()
    
    from app.indexing.metadata_store.repository import LegalRepository
    from app.indexing.metadata_store.database import Database
    from app.retrieval.pipeline import RetrievalPipeline
    from app.retrieval.query.analyzer import QueryAnalyzer
    from app.retrieval.filters.metadata_filter import MetadataFilter
    from app.retrieval.dense.retriever import DenseRetriever
    from app.retrieval.lexical.retriever import BM25Retriever
    from app.retrieval.fusion.rrf import RRFFusion
    from app.retrieval.context.parent_expander import ParentContextExpander
    from app.retrieval.context.context_builder import LegalContextBuilder
    from app.retrieval.active_index import ActiveIndex
    from app.indexing.embeddings.factory import EmbeddingModelFactory
    from app.retrieval.reranking.factory import RerankerFactory
    
    active = ActiveIndex(settings.index_root_dir)
    database = Database(active.sqlite_path)
    database.initialize()
    repository = LegalRepository(database)
    
    print("Loading embedding model...")
    embedding = EmbeddingModelFactory.create(
        provider="sentence_transformers",
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        local_files_only=settings.model_local_files_only,
    )
    embedding.load()
    
    print("Loading reranker model...")
    reranker = RerankerFactory.create(
        provider="local_transformers",
        model_name=settings.reranker_model_name,
        device=settings.reranker_device,
        local_files_only=settings.model_local_files_only,
        trust_remote_code=settings.model_trust_remote_code,
        repository=repository,
        parameter_budget_approved=False,
    )
    reranker.load()
    
    verify_budget = getattr(reranker, "verify_parameter_budget", None)
    if callable(verify_budget):
        verify_budget(3600000000)
        
    parent_expander = ParentContextExpander(
        repository=repository,
        neighbor_window=settings.context_neighbor_window,
        same_parent_only=True,
        deduplicate_overlap=True,
    )
    context_builder = LegalContextBuilder(
        max_tokens=settings.context_max_tokens,
    )
    
    pipeline = RetrievalPipeline(
        query_analyzer=QueryAnalyzer(),
        metadata_filter=MetadataFilter(repository, enabled=settings.metadata_filter_enabled),
        dense_retriever=DenseRetriever(embedding, repository=repository, index_root=settings.index_root_dir),
        bm25_retriever=BM25Retriever(repository=repository, index_root=settings.index_root_dir),
        fusion=RRFFusion(),
        reranker=reranker,
        parent_expander=parent_expander,
        context_builder=context_builder,
        trace=True
    )
    
    queries = [
        "Thủ tục cấp giấy phép hoạt động đối với cơ sở dịch vụ thẩm mỹ?",
        "Thời hạn giải quyết việc đăng ký hoạt động đối với văn phòng luật sư?",
        "Quyền lợi của người lao động khi bị đơn phương chấm dứt hợp đồng?",
        "Mức phạt vi phạm quy định về thời giờ làm việc của lao động nữ?",
        "Trình tự thủ tục đình công theo quy định mới nhất?"
    ]
    
    print("\nWarmup query...")
    pipeline.retrieve(queries[0])
    
    print("\nRunning test queries...")
    start_t = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        res = pipeline.retrieve(q)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"Query: {q[:30]}... | Latency: {dt:.1f}ms | Candidates: {len(res.candidates)} | Evidences: {len(res.evidences)}")
    total_dt = (time.perf_counter() - start_t) * 1000.0
    print(f"\nTotal time for {len(queries)} queries: {total_dt:.1f}ms (Avg: {total_dt/len(queries):.1f}ms/query)")

if __name__ == "__main__":
    main()
