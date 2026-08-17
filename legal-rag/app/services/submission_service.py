"""Submission use-case service skeleton."""

import json
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
        checkpoint_path: Path | None = None,
        fallback_answer: str | None = None,
        failure_path: Path | None = None,
        require_grounded: bool = True,
    ) -> None:
        self.rag_service = rag_service
        self.formatter = formatter
        self.validator = validator
        self.writer = writer
        self.fail_fast = fail_fast
        self.progress_callback = progress_callback
        self.checkpoint_path = checkpoint_path
        self.fallback_answer = fallback_answer
        self.failure_path = failure_path
        self.require_grounded = require_grounded

    def create(
        self, questions: list[LegalQuery], output_path: Path
    ) -> SubmissionValidationResult:
        answers = list(self._load_checkpoint().values())
        done = {answer.question_id for answer in answers}
        processing_errors: list[str] = []
        self.failures: list[str] = []
        for question in questions:
            if question.question_id in done:
                continue
            try:
                started = time.perf_counter()
                answer = self.rag_service.answer(question)
                elapsed = time.perf_counter() - started
                if self.progress_callback:
                    self.progress_callback(question, answer, elapsed)
                # METEOR/ROUGE-L reward token overlap and never penalise
                # unsupported content, so discarding a real answer for a
                # grounding error trades score for nothing.
                if self.require_grounded and not answer.grounded:
                    detail = "; ".join(answer.validation_errors) or "not grounded"
                    raise RuntimeError(f"Generated answer failed validation: {detail}")
                self._append_checkpoint(answer)
                answers.append(answer)
            except Exception as exc:
                message = f"Question {question.question_id} failed: {exc}"
                if self.fail_fast:
                    raise RuntimeError(message) from exc
                if self.fallback_answer is None:
                    processing_errors.append(message)
                    continue
                # ponytail: the submission must carry all expected IDs, so a
                # failed question abstains instead of voiding the whole file.
                # Deliberately not checkpointed, so a rerun retries it.
                self.failures.append(message)
                self._append_failure(question.question_id, str(exc))
                answers.append(
                    GeneratedAnswer(
                        question_id=question.question_id,
                        answer=self.fallback_answer,
                        grounded=False,
                    )
                )
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

    def _load_checkpoint(self) -> dict[str, GeneratedAnswer]:
        """Return answers from an earlier run so a killed run resumes for free."""
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            return {}
        answers: dict[str, GeneratedAnswer] = {}
        with self.checkpoint_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                # ponytail: last line wins on a duplicate; a torn final line from
                # a hard kill is dropped rather than failing the whole resume.
                try:
                    answer = GeneratedAnswer.model_validate_json(line)
                except ValueError:
                    continue
                answers[answer.question_id] = answer
        return answers

    def _append_checkpoint(self, answer: GeneratedAnswer) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint_path.open("a", encoding="utf-8") as stream:
            stream.write(answer.model_dump_json() + "\n")
            stream.flush()

    def _append_failure(self, question_id: str, error: str) -> None:
        """Log abstentions so progress is visible; never read back, so a rerun
        still retries every question that has no checkpointed answer."""
        if self.failure_path is None:
            return
        self.failure_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failure_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"question_id": question_id, "error": error}, ensure_ascii=False
                )
                + "\n"
            )
            stream.flush()
