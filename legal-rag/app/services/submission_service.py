"""Submission use-case service skeleton."""

from pathlib import Path

from app.core.exceptions import SubmissionValidationError
from app.domain.queries import LegalQuery
from app.domain.submission import SubmissionValidationResult
from app.services.legal_rag_service import LegalRAGService
from app.submission.formatter import SubmissionFormatter
from app.submission.json_io import load_json_strict
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
        """Answer every unique question and write a verified submission file."""
        expected_ids = {question.question_id for question in questions}
        if len(expected_ids) != len(questions):
            raise SubmissionValidationError("Question input contains duplicate IDs.")

        answers = [self.rag_service.answer(question) for question in questions]
        submission = self.formatter.format(answers)
        payload = {
            question_id: answer.model_dump()
            for question_id, answer in submission.items()
        }
        result = self.validator.validate(payload, expected_ids)
        if not result.valid:
            raise SubmissionValidationError("; ".join(result.errors))

        self.writer.write(submission, output_path)
        persisted = load_json_strict(output_path)
        persisted_result = self.validator.validate(persisted, expected_ids)
        if not persisted_result.valid:
            raise SubmissionValidationError(
                "Written submission failed validation: "
                + "; ".join(persisted_result.errors)
            )
        return persisted_result
