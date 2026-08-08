"""End-to-end legal question-answering service."""

from collections.abc import Iterator

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
        context = self.retrieval_pipeline.retrieve(query)
        try:
            return self.generation_pipeline.generate(query, context)
        finally:
            self.generation_pipeline.unload()

    def answer_many(self, queries: list[LegalQuery]) -> Iterator[GeneratedAnswer]:
        """Answer a batch while loading each neural model only once."""
        contexts = self.retrieval_pipeline.retrieve_many(queries)
        try:
            for query, context in zip(queries, contexts, strict=True):
                yield self.generation_pipeline.generate(query, context)
        finally:
            self.generation_pipeline.unload()
