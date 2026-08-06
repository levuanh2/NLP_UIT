"""Submission formatting and writing tests."""

import json
from pathlib import Path

from app.domain.generation import GeneratedAnswer
from app.submission.formatter import SubmissionFormatter
from app.submission.writer import SubmissionWriter


def test_submission_formatter_uses_question_id_as_key() -> None:
    # Arrange
    answer = GeneratedAnswer(
        question_id="147194",
        answer="Câu trả lời.",
        grounded=None,
        confidence=None,
    )

    # Act
    submission = SubmissionFormatter().format([answer])

    # Assert
    assert set(submission) == {"147194"}


def test_submission_formatter_contains_only_answer() -> None:
    # Arrange
    answer = GeneratedAnswer(
        question_id="147194",
        answer="Theo Điều 1...",
        evidence_ids=["e-1"],
        grounded=True,
        confidence=None,
    )

    # Act
    payload = SubmissionFormatter().format([answer])["147194"].model_dump()

    # Assert
    assert payload == {"answer": "Theo Điều 1..."}


def test_submission_writer_uses_utf8(tmp_path: Path) -> None:
    # Arrange
    submission = SubmissionFormatter().format(
        [
            GeneratedAnswer(
                question_id="1",
                answer="Quyền và nghĩa vụ",
                grounded=None,
                confidence=None,
            )
        ]
    )
    output = tmp_path / "submission.json"

    # Act
    SubmissionWriter().write(submission, output)

    # Assert
    assert output.read_bytes().decode("utf-8")


def test_submission_writer_preserves_vietnamese_characters(tmp_path: Path) -> None:
    # Arrange
    expected = "Người lao động có quyền."
    submission = SubmissionFormatter().format(
        [
            GeneratedAnswer(
                question_id="147195",
                answer=expected,
                grounded=None,
                confidence=None,
            )
        ]
    )
    output = tmp_path / "submission.json"

    # Act
    SubmissionWriter().write(submission, output)

    # Assert
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["147195"]["answer"] == expected
    assert "Người lao động" in output.read_text(encoding="utf-8")
