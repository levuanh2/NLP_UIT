"""End-to-end legal question-answering service skeleton."""

from app.domain.generation import GeneratedAnswer, GenerationRequest
from app.domain.queries import LegalQuery
from app.generation.pipeline import GenerationPipeline
from app.retrieval.pipeline import RetrievalPipeline


class LegalRAGService:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        generation_pipeline: GenerationPipeline,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.generation_pipeline = generation_pipeline

    def answer_batch(self, queries: list[LegalQuery]) -> list[GeneratedAnswer]:
        """Retrieve for each query, then answer them all in one decode pass."""
        requests = [
            GenerationRequest(
                question_id=query.question_id,
                question=query.question,
                retrieval_result=self.retrieval_pipeline.retrieve(query),
            )
            for query in queries
        ]
        return self.generation_pipeline.generate_batch(requests)

    def answer(self, query: LegalQuery) -> GeneratedAnswer:
        """Retrieve legal evidence and generate a grounded answer."""
        retrieval_result = self.retrieval_pipeline.retrieve(query)
        return self.generation_pipeline.generate(
            GenerationRequest(
                question_id=query.question_id,
                question=query.question,
                retrieval_result=retrieval_result,
            )
        )
