"""Evaluation runner skeleton."""

from app.domain.evaluation import EvaluationReport, EvaluationSample
from app.services.legal_rag_service import LegalRAGService


class EvaluationRunner:
    def __init__(self, rag_service: LegalRAGService) -> None:
        self.rag_service = rag_service

    def run(self, samples: list[EvaluationSample]) -> EvaluationReport:
        # TODO(phase-implementation):
        # Execute local evaluation and aggregate configured metrics.
        raise NotImplementedError
