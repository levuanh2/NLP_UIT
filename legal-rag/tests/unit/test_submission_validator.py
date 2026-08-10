"""Submission schema validator tests."""

import json
from pathlib import Path

from app.submission.validator import SubmissionValidator


def test_submission_validator_rejects_missing_answer() -> None:
    result = SubmissionValidator().validate({"1": {}}, {"1"})
    assert not result.valid


def test_submission_validator_rejects_non_string_answer() -> None:
    result = SubmissionValidator().validate({"1": {"answer": 42}}, {"1"})
    assert not result.valid


def test_submission_validator_rejects_missing_question() -> None:
    result = SubmissionValidator().validate({}, {"1"})
    assert result.missing_question_ids == ["1"]
    assert not result.valid


def test_submission_validator_rejects_extra_question() -> None:
    result = SubmissionValidator().validate(
        {"1": {"answer": "A"}, "2": {"answer": "B"}}, {"1"}
    )
    assert result.unexpected_question_ids == ["2"]
    assert not result.valid


def test_submission_validator(tmp_path: Path) -> None:
    path = tmp_path / "submission.json"
    path.write_text(
        json.dumps({"1": {"answer": "Câu trả lời"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = SubmissionValidator().validate(path, {"1"})

    assert result.valid is True


def test_submission_validator_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "submission.json"
    path.write_text(
        '{"1":{"answer":"A"},"1":{"answer":"B"}}', encoding="utf-8"
    )

    result = SubmissionValidator().validate(path, {"1"})

    assert result.valid is False
    assert "Duplicate JSON key" in result.errors[0]
