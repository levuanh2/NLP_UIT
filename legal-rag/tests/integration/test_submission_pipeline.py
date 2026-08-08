"""Submission pipeline integration tests."""

import json
from pathlib import Path

from app.domain.generation import GeneratedAnswer
from app.domain.queries import LegalQuery
from app.services.submission_service import SubmissionService
from app.submission.formatter import SubmissionFormatter
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


class _RAG:
    def answer(self, query: LegalQuery) -> GeneratedAnswer:
        return GeneratedAnswer(
            question_id=query.question_id,
            answer="Câu trả lời.",
            grounded=True,
            confidence=1.0,
        )


def test_submission_pipeline_writes_one_answer_per_question(tmp_path: Path) -> None:
    service = SubmissionService(
        _RAG(),
        SubmissionFormatter(),
        SubmissionValidator(),
        SubmissionWriter(),  # type: ignore[arg-type]
    )
    output = tmp_path / "submission.json"
    result = service.create([LegalQuery(question_id="1", question="q")], output)
    assert result.valid
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "1": {"answer": "Câu trả lời."}
    }
