"""Submission formatting and writing tests."""

import json
from pathlib import Path

import pytest

from app.core.exceptions import SubmissionValidationError
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


def test_internal_result_formatter_strips_debug_fields() -> None:
    # Arrange
    internal_results = [
        {
            "question_id": "147194",
            "answer": "Theo Điều 37...",
            "citations": ["Điều 37"],
            "evidence_ids": ["e-1"],
            "confidence": "high",
            "reasoning": "internal only",
        }
    ]

    # Act
    payload = (
        SubmissionFormatter()
        .format_internal_results(internal_results)["147194"]
        .model_dump()
    )

    # Assert
    assert payload == {"answer": "Theo Điều 37..."}


def test_internal_result_formatter_rejects_duplicate_ids() -> None:
    # Arrange
    internal_results = [
        {"question_id": "1", "answer": "A"},
        {"question_id": "1", "answer": "B"},
    ]

    # Act / Assert
    with pytest.raises(ValueError, match="Duplicate question_id"):
        SubmissionFormatter().format_internal_results(internal_results)


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


def test_submission_writer_requires_exact_filename(tmp_path: Path) -> None:
    submission = SubmissionFormatter().format(
        [
            GeneratedAnswer(
                question_id="1",
                answer="Câu trả lời.",
                grounded=None,
                confidence=None,
            )
        ]
    )

    with pytest.raises(SubmissionValidationError, match="submission.json"):
        SubmissionWriter().write(submission, tmp_path / "result.json")
