"""End-to-end legal question-answering service skeleton."""

from app.domain.generation import GeneratedAnswer
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

    def answer(self, query: LegalQuery) -> GeneratedAnswer:
        """Retrieve legal evidence and generate a grounded answer."""
        # TODO(phase-implementation):
        # Implement retrieval-to-generation orchestration.
        raise NotImplementedError
