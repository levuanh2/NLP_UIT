"""Submission use-case service skeleton."""

import time
from collections.abc import Callable
from pathlib import Path

from app.domain.generation import GeneratedAnswer
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
        fail_fast: bool = False,
        progress_callback: (
            Callable[[LegalQuery, GeneratedAnswer, float], None] | None
        ) = None,
    ) -> None:
        self.rag_service = rag_service
        self.formatter = formatter
        self.validator = validator
        self.writer = writer
        self.fail_fast = fail_fast
        self.progress_callback = progress_callback

    def create(
        self, questions: list[LegalQuery], output_path: Path
    ) -> SubmissionValidationResult:
        answers = []
        processing_errors: list[str] = []
        for question in questions:
            try:
                started = time.perf_counter()
                answer = self.rag_service.answer(question)
                elapsed = time.perf_counter() - started
                if self.progress_callback:
                    self.progress_callback(question, answer, elapsed)
                if not answer.grounded:
                    detail = "; ".join(answer.validation_errors) or "not grounded"
                    raise RuntimeError(f"Generated answer failed validation: {detail}")
                answers.append(answer)
            except Exception as exc:
                message = f"Question {question.question_id} failed: {exc}"
                if self.fail_fast:
                    raise RuntimeError(message) from exc
                processing_errors.append(message)
        submission = self.formatter.format(answers)
        payload = {
            question_id: answer.model_dump()
            for question_id, answer in submission.items()
        }
        expected_ids = {question.question_id for question in questions}
        validation = self.validator.validate(payload, expected_ids)
        errors = processing_errors + validation.errors
        result = validation.model_copy(update={"valid": not errors, "errors": errors})
        if result.valid:
            self.writer.write(
                submission,
                output_path,
                expected_question_ids=expected_ids,
            )
        return result
