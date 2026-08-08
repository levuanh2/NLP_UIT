"""Offline evaluation runner."""

from app.domain.evaluation import EvaluationReport, EvaluationSample
from app.domain.queries import LegalQuery
from app.evaluation.citation_metrics import citation_precision
from app.evaluation.generation_metrics import answer_similarity
from app.evaluation.retrieval_metrics import mean_reciprocal_rank, recall_at_k
from app.services.legal_rag_service import LegalRAGService


class EvaluationRunner:
    def __init__(self, rag_service: LegalRAGService) -> None:
        self.rag_service = rag_service

    def run(self, samples: list[EvaluationSample]) -> EvaluationReport:
        if not samples:
            return EvaluationReport()
        generation: list[float] = []
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        citations: list[float] = []
        grounded: list[float] = []
        for sample in samples:
            answer = self.rag_service.answer(
                LegalQuery(question_id=sample.question_id, question=sample.question)
            )
            relevant = set(sample.relevant_child_ids)
            if sample.expected_answer:
                generation.append(
                    answer_similarity(answer.answer, sample.expected_answer)
                )
            if relevant:
                recalls.append(recall_at_k(answer.evidence_ids, relevant, 5))
                reciprocal_ranks.append(
                    mean_reciprocal_rank(answer.evidence_ids, relevant)
                )
                citations.append(citation_precision(answer.citations, relevant))
            grounded.append(float(bool(answer.grounded)))
        metrics = {
            "meteor": _mean(generation),
            "recall_at_5": _mean(recalls),
            "mrr": _mean(reciprocal_ranks),
            "citation_precision": _mean(citations),
            "grounded_rate": _mean(grounded),
        }
        return EvaluationReport(metrics=metrics, sample_count=len(samples))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
