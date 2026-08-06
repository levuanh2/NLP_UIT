"""Submission use-case service skeleton."""

from pathlib import Path

from app.domain.queries import LegalQuery
from app.domain.submission import SubmissionValidationResult
from app.services.legal_rag_service import LegalRAGService
from app.submission.formatter import SubmissionFormatter
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


class SubmissionService:
    def __init__(
        self,
        rag_service: LegalRAGService,
        formatter: SubmissionFormatter,
        validator: SubmissionValidator,
        writer: SubmissionWriter,
    ) -> None:
        self.rag_service = rag_service
        self.formatter = formatter
        self.validator = validator
        self.writer = writer

    def create(
        self, questions: list[LegalQuery], output_path: Path
    ) -> SubmissionValidationResult:
        # TODO(phase-implementation):
        # Answer each real question, validate the payload, then write it.
        raise NotImplementedError
