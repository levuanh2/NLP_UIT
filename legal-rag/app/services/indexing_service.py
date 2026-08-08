"""Index-building use-case service."""

from app.indexing.pipeline import IndexingPipeline, IndexingResult
from app.ingestion.pipeline import IngestionResult


class IndexingService:
    def __init__(self, pipeline: IndexingPipeline) -> None:
        self.pipeline = pipeline

    def build(self, ingestion_results: list[IngestionResult]) -> IndexingResult:
        parents = [
            chunk for result in ingestion_results for chunk in result.parent_chunks
        ]
        children = [
            chunk for result in ingestion_results for chunk in result.child_chunks
        ]
        if not children:
            raise ValueError("Ingestion produced no child chunks to index.")
        return self.pipeline.build(children, parents)
