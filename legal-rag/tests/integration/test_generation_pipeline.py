"""Generation integration behavior not requiring real model weights."""

from app.domain.generation import GenerationRequest
from app.domain.queries import QueryMetadata
from app.domain.retrieval import RetrievalResult
from app.generation.citation_validator import CitationValidator
from app.generation.grounding import GroundingValidator
from app.generation.pipeline import GenerationPipeline
from app.generation.prompt import LegalPromptBuilder
from app.generation.validation.abstention_validator import AbstentionValidator
from tests.unit.test_generation import MockLocalLLM


def test_generation_pipeline_abstains_on_insufficient_evidence() -> None:
    llm = MockLocalLLM("must not run")
    citations = CitationValidator()
    pipeline = GenerationPipeline(
        llm,
        LegalPromptBuilder(),
        citations,
        GroundingValidator(citations),
        AbstentionValidator(),
    )
    retrieval = RetrievalResult(
        query="Câu hỏi",
        query_metadata=QueryMetadata(raw_query="Câu hỏi"),
        candidates=[],
        evidences=[],
        active_index_version="fixture",
        dense_count=0,
        bm25_count=0,
        fused_count=0,
        reranked_count=0,
    )

    answer = pipeline.generate(
        GenerationRequest(
            question_id="q", question="Câu hỏi", retrieval_result=retrieval
        )
    )

    assert answer.abstained is True
    assert answer.grounded is True
    assert llm.calls == 0
