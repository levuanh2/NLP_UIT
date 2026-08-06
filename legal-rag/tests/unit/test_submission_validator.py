"""Submission schema validator tests."""

from app.submission.validator import SubmissionValidator


def test_submission_validator_rejects_missing_answer() -> None:
    # Arrange / Act
    result = SubmissionValidator().validate({"1": {}}, {"1"})

    # Assert
    assert not result.valid


def test_submission_validator_rejects_non_string_answer() -> None:
    # Arrange / Act
    result = SubmissionValidator().validate({"1": {"answer": 42}}, {"1"})

    # Assert
    assert not result.valid


def test_submission_validator_rejects_missing_question() -> None:
    # Arrange / Act
    result = SubmissionValidator().validate({}, {"1"})

    # Assert
    assert result.missing_question_ids == ["1"]
    assert not result.valid


def test_submission_validator_rejects_extra_question() -> None:
    # Arrange / Act
    result = SubmissionValidator().validate(
        {"1": {"answer": "A"}, "2": {"answer": "B"}}, {"1"}
    )

    # Assert
    assert result.unexpected_question_ids == ["2"]
    assert not result.valid
