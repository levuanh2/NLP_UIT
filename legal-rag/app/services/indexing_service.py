"""Index-building use-case service skeleton."""

from app.indexing.pipeline import IndexingPipeline, IndexingResult
from app.ingestion.pipeline import IngestionResult


class IndexingService:
    def __init__(self, pipeline: IndexingPipeline) -> None:
        self.pipeline = pipeline

    def build(self, ingestion_results: list[IngestionResult]) -> IndexingResult:
        # TODO(phase-implementation):
        # Collect validated chunks and invoke the indexing pipeline.
        raise NotImplementedError
