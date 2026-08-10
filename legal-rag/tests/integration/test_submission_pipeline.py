"""Submission orchestration integration tests."""

import json
from pathlib import Path

from app.domain.generation import GeneratedAnswer
from app.domain.queries import LegalQuery
from app.services.submission_service import SubmissionService
from app.submission.formatter import SubmissionFormatter
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


class FakeRAGService:
    def answer(self, query: LegalQuery) -> GeneratedAnswer:
        return GeneratedAnswer(
            question_id=query.question_id,
            answer=f"Trả lời có căn cứ cho {query.question_id} [1].",
            grounded=True,
        )


def test_submission_pipeline_writes_one_answer_per_question(tmp_path: Path) -> None:
    questions = [
        LegalQuery(question_id="1", question="Câu hỏi một"),
        LegalQuery(question_id="2", question="Câu hỏi hai"),
    ]
    output = tmp_path / "submission.json"
    service = SubmissionService(
        FakeRAGService(),  # type: ignore[arg-type]
        SubmissionFormatter(),
        SubmissionValidator(),
        SubmissionWriter(),
    )

    result = service.create(questions, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.valid is True
    assert set(payload) == {"1", "2"}
    assert all(set(value) == {"answer"} for value in payload.values())
