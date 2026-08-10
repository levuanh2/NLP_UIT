"""Small, local end-to-end retrieval fixture (five documents only)."""

from pathlib import Path

import numpy as np
import pytest

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument
from app.domain.generation import GenerationRequest
from app.domain.retrieval import RetrievalCandidate
from app.generation.citation_validator import CitationValidator
from app.generation.grounding import GroundingValidator
from app.generation.llm.base import BaseLLMGenerator
from app.generation.pipeline import GenerationPipeline
from app.generation.prompt import LegalPromptBuilder
from app.generation.validation.abstention_validator import AbstentionValidator
from app.indexing.embeddings.base import BaseEmbeddingModel
from app.indexing.lexical.bm25_index import BM25IndexWriter
from app.indexing.manifest import IndexManifest
from app.indexing.metadata_store.database import Database
from app.indexing.metadata_store.repository import LegalRepository
from app.indexing.vector_store.writer import FAISSShardWriter
from app.indexing.versioning import IndexVersionManager
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.context.context_builder import LegalContextBuilder
from app.retrieval.context.parent_expander import ParentContextExpander
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.fusion.rrf import RRFFusion
from app.retrieval.metadata_filter import MetadataFilter
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.reranking.base import BaseReranker


class FixtureEmbedding(BaseEmbeddingModel):
    model_name = "fixture/legal-embedding"
    terms = ("doanh", "hình", "lao", "đất")

    def load(self) -> None:
        pass

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def dimension(self) -> int:
        return len(self.terms)

    def unload(self) -> None:
        pass

    def _vector(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vector = np.asarray(
            [lowered.count(term) for term in self.terms], dtype=np.float32
        )
        if not vector.any():
            vector[0] = 0.01
        return vector


class TokenOverlapReranker(BaseReranker):
    """Test adapter with a real lexical relevance calculation."""

    def __init__(self, repository: LegalRepository) -> None:
        self.repository = repository

    def load(self) -> None:
        pass

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        query_terms = set(query.lower().split())
        scored: list[tuple[RetrievalCandidate, float]] = []
        for candidate in candidates:
            child = self.repository.get_child(candidate.child_id)
            assert child is not None
            score = len(query_terms.intersection(child.text.lower().split()))
            scored.append((candidate, float(score)))
        scored.sort(key=lambda item: (-item[1], item[0].child_id))
        return [
            item.model_copy(
                update={
                    "score": score,
                    "source": "reranker",
                    "rank": rank,
                    "rerank_score": score,
                }
            )
            for rank, (item, score) in enumerate(scored[:top_k], start=1)
        ]

    def unload(self) -> None:
        pass


class MockGenerationLLM(BaseLLMGenerator):
    model_name = "mock/local-generation"

    def load(self) -> None:
        pass

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        assert "Luật Doanh nghiệp" in prompt
        return "Doanh nghiệp phải đáp ứng điều kiện đăng ký [1]."

    def unload(self) -> None:
        pass


def _parent(document_id: int, article: str, text: str) -> ParentChunk:
    return ParentChunk(
        parent_id=f"p-{document_id}-{article.split()[-1]}",
        document_id=document_id,
        document_name=(
            "Luật Doanh nghiệp" if document_id == 1 else f"Luật Mẫu {document_id}"
        ),
        source_link=f"https://example.test/{document_id}",
        chapter="Chương I",
        section=None,
        article=article,
        position=0,
        text=text,
        token_count=len(text.split()),
    )


def _children(parent: ParentChunk, texts: list[str]) -> list[ChildChunk]:
    children: list[ChildChunk] = []
    for position, text in enumerate(texts):
        child_id = f"{parent.parent_id}-c-{position}"
        children.append(
            ChildChunk(
                child_id=child_id,
                parent_id=parent.parent_id,
                document_id=parent.document_id,
                document_name=parent.document_name,
                source_link=parent.source_link,
                chapter=parent.chapter,
                section=parent.section,
                article=parent.article,
                clause=f"Khoản {position + 1}",
                point="Điểm a" if position == 1 else None,
                position=position,
                previous_child_id=(
                    f"{parent.parent_id}-c-{position - 1}" if position else None
                ),
                next_child_id=(
                    f"{parent.parent_id}-c-{position + 1}"
                    if position < len(texts) - 1
                    else None
                ),
                text=text,
                embedding_text=text,
                token_count=len(text.split()),
            )
        )
    return children


@pytest.fixture
def retrieval_fixture(
    tmp_path: Path,
) -> tuple[Path, LegalRepository, FixtureEmbedding]:
    root = tmp_path / "indexes"
    version = "v9"
    manager = IndexVersionManager(root)
    version_dir = manager.ensure_version(version)
    database = Database(version_dir / "metadata" / "legal.sqlite")
    database.initialize()
    repository = LegalRepository(database)
    specifications = [
        (
            1,
            "Điều 37",
            [
                "Doanh nghiệp chuẩn bị hồ sơ đăng ký",
                "Doanh nghiệp đáp ứng điều kiện đăng ký hợp lệ",
                "Cơ quan cấp giấy cho doanh nghiệp",
            ],
        ),
        (2, "Điều 37", ["Hình sự quy định trách nhiệm và hình phạt"]),
        (3, "Điều 12", ["Lao động ký hợp đồng và nhận tiền lương"]),
        (4, "Điều 8", ["Đất đai được đăng ký quyền sử dụng"]),
        (5, "Điều 4", ["Hình thức doanh nghiệp xã hội được công nhận"]),
    ]
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    for document_id, article, texts in specifications:
        document_name = (
            "Luật Doanh nghiệp" if document_id == 1 else f"Luật Mẫu {document_id}"
        )
        repository.save_document(
            LegalDocument(
                document_id=document_id,
                document_name=document_name,
                source_link=f"https://example.test/{document_id}",
                raw_text=" ".join(texts),
            ),
            checksum=f"{document_id:064d}",
            chunking_version="fixture-v1",
            index_version=version,
            status="completed",
            embedding_model=FixtureEmbedding.model_name,
            embedding_dimension=4,
        )
        parent = _parent(document_id, article, " ".join(texts))
        document_children = _children(parent, texts)
        repository.save_chunks([parent], document_children)
        parents.append(parent)
        children.extend(document_children)

    embedding = FixtureEmbedding()
    vector_writer = FAISSShardWriter(
        version_dir / "faiss", 4, index_type="flat", shard_size=3
    )
    vector_writer.add_batch(
        [child.child_id for child in children],
        embedding.embed_documents([child.embedding_text for child in children]),
    )
    vector_writer.finalize()
    bm25_writer = BM25IndexWriter(version_dir / "bm25")
    bm25_writer.add_batch(children)
    bm25_writer.close()
    manager.write_manifest(
        IndexManifest(
            index_version=version,
            corpus_path="fixture://five-documents",
            document_count=5,
            valid_document_count=5,
            failed_document_count=0,
            parent_count=len(parents),
            child_count=len(children),
            embedding_model=FixtureEmbedding.model_name,
            embedding_dimension=4,
            chunking_version="fixture-v1",
            faiss_index_type="flat",
            status="ready",
        )
    )
    manager.publish(version)
    return root, repository, embedding


def _pipeline(
    root: Path,
    repository: LegalRepository,
    embedding: FixtureEmbedding,
    *,
    max_tokens: int = 100,
) -> RetrievalPipeline:
    return RetrievalPipeline(
        QueryAnalyzer(),
        MetadataFilter(repository),
        DenseRetriever(embedding, index_root=root),
        BM25Retriever(index_root=root),
        RRFFusion(),
        TokenOverlapReranker(repository),
        ParentContextExpander(repository, neighbor_window=1),
        LegalContextBuilder(max_tokens=max_tokens),
        reranker_top_k=5,
    )


def test_dense_retriever_uses_current_index(retrieval_fixture) -> None:
    root, _, embedding = retrieval_fixture
    retriever = DenseRetriever(embedding, index_root=root)

    results = retriever.retrieve("doanh nghiệp", top_k=3)

    assert retriever.active_index_version == "v9"
    assert results
    assert all(item.source == "dense" for item in results)


def test_bm25_retriever_uses_current_index(retrieval_fixture) -> None:
    root, _, _ = retrieval_fixture
    retriever = BM25Retriever(index_root=root)

    results = retriever.retrieve("đăng ký doanh nghiệp", top_k=3)

    assert retriever.active_index_version == "v9"
    assert results
    assert all(item.source == "bm25" for item in results)


def test_retrieval_pipeline(retrieval_fixture) -> None:
    root, repository, embedding = retrieval_fixture
    pipeline = _pipeline(root, repository, embedding)

    result = pipeline.retrieve(
        "Theo Điều 37 Luật Doanh nghiệp, điều kiện đăng ký là gì?"
    )

    assert result.active_index_version == "v9"
    assert result.metadata_filter_applied is True
    assert result.query_metadata.document_name == "Luật Doanh nghiệp"
    assert result.dense_count > 0 and result.bm25_count > 0
    assert result.fused_count >= result.reranked_count > 0
    assert all(item.child_id.startswith("p-1-37") for item in result.candidates)


def test_metadata_filter_no_match_fallback(retrieval_fixture) -> None:
    root, repository, embedding = retrieval_fixture
    pipeline = _pipeline(root, repository, embedding)

    result = pipeline.retrieve(
        "Theo Điều 999 Luật Doanh nghiệp, đăng ký doanh nghiệp thế nào?"
    )

    assert result.metadata_filter_fallback is True
    assert result.metadata_filter_applied is False
    assert result.candidates


def test_parent_expansion_after_rerank(retrieval_fixture) -> None:
    root, repository, embedding = retrieval_fixture
    result = _pipeline(root, repository, embedding).retrieve(
        "Theo Điều 37 Luật Doanh nghiệp, điều kiện đăng ký là gì?"
    )

    evidence = result.evidences[0]
    assert evidence.parent_id == "p-1-37"
    assert "chuẩn bị hồ sơ" in evidence.text
    assert "đáp ứng điều kiện" in evidence.text
    assert evidence.document_name == "Luật Doanh nghiệp"


def test_context_budget_after_retrieval(retrieval_fixture) -> None:
    root, repository, embedding = retrieval_fixture
    result = _pipeline(root, repository, embedding, max_tokens=7).retrieve(
        "Theo Điều 37 Luật Doanh nghiệp, điều kiện đăng ký là gì?"
    )

    assert result.evidences
    assert sum(len(item.text.split()) for item in result.evidences) <= 7


def test_retrieval_to_grounded_generation(retrieval_fixture) -> None:
    root, repository, embedding = retrieval_fixture
    retrieval_result = _pipeline(root, repository, embedding).retrieve(
        "Theo Điều 37 Luật Doanh nghiệp, điều kiện đăng ký là gì?"
    )
    citations = CitationValidator()
    generation = GenerationPipeline(
        MockGenerationLLM(),
        LegalPromptBuilder(max_context_tokens=1000, reserved_generation_tokens=64),
        citations,
        GroundingValidator(citations),
        AbstentionValidator(),
        max_new_tokens=64,
    )

    answer = generation.generate(
        GenerationRequest(
            question_id="fixture-question",
            question=retrieval_result.query,
            retrieval_result=retrieval_result,
        )
    )

    assert answer.grounded is True
    assert answer.citations[0].document_id == 1
    assert answer.citations[0].article == "Điều 37"
