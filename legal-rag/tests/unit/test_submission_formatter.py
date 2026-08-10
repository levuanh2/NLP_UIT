"""Submission formatting and writing tests."""

import json
from pathlib import Path

from app.domain.generation import GeneratedAnswer
from app.submission.formatter import SubmissionFormatter
from app.submission.writer import SubmissionWriter


def test_submission_formatter_uses_question_id_as_key() -> None:
    answer = GeneratedAnswer(
        question_id="147194", answer="Câu trả lời.", grounded=False
    )

    submission = SubmissionFormatter().format([answer])

    assert set(submission) == {"147194"}


def test_submission_formatter_contains_only_answer() -> None:
    answer = GeneratedAnswer(
        question_id="147194",
        answer="Theo Điều 1...",
        evidence_ids=["e-1"],
        grounded=True,
    )

    payload = SubmissionFormatter().format([answer])["147194"].model_dump()

    assert payload == {"answer": "Theo Điều 1..."}


def test_submission_writer_uses_utf8(tmp_path: Path) -> None:
    submission = SubmissionFormatter().format(
        [GeneratedAnswer(question_id="1", answer="Quyền và nghĩa vụ", grounded=True)]
    )
    output = tmp_path / "submission.json"

    SubmissionWriter().write(submission, output)

    assert output.read_bytes().decode("utf-8")


def test_submission_writer_preserves_vietnamese_characters(tmp_path: Path) -> None:
    expected = "Người lao động có quyền."
    submission = SubmissionFormatter().format(
        [GeneratedAnswer(question_id="147195", answer=expected, grounded=True)]
    )
    output = tmp_path / "submission.json"

    SubmissionWriter().write(submission, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["147195"]["answer"] == expected
    assert "Người lao động" in output.read_text(encoding="utf-8")


def test_submission_writer(tmp_path: Path) -> None:
    output = tmp_path / "submission.json"
    answers = {
        "147194": GeneratedAnswer(
            question_id="147194",
            answer="Câu trả lời có căn cứ [1].",
            grounded=True,
        )
    }

    SubmissionWriter().write(answers, output, expected_question_ids={"147194"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "147194": {"answer": "Câu trả lời có căn cứ [1]."}
    }
