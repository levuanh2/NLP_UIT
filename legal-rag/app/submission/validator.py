"""Strict in-memory and UTF-8 JSON submission validation."""

import json
from pathlib import Path
from typing import Any

from app.domain.submission import SubmissionValidationResult


class _DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


class SubmissionValidator:
    def validate(
        self,
        source: dict[str, Any] | Path,
        expected_question_ids: set[str],
    ) -> SubmissionValidationResult:
        if isinstance(source, Path):
            try:
                with source.open(encoding="utf-8") as stream:
                    submission: Any = json.load(
                        stream, object_pairs_hook=_unique_object
                    )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                _DuplicateKeyError,
            ) as exc:
                return SubmissionValidationResult(
                    valid=False,
                    errors=[f"Submission is not valid UTF-8 JSON: {exc}"],
                    missing_question_ids=sorted(expected_question_ids),
                    unexpected_question_ids=[],
                )
        else:
            submission = source
        errors: list[str] = []
        if not isinstance(submission, dict):
            return SubmissionValidationResult(
                valid=False,
                errors=["Submission root must be a JSON object."],
                missing_question_ids=sorted(expected_question_ids),
                unexpected_question_ids=[],
            )
        if any(not isinstance(question_id, str) for question_id in submission):
            errors.append("Every question ID must be a string.")
        actual_ids = {key for key in submission if isinstance(key, str)}
        missing = sorted(expected_question_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_question_ids)
        if missing:
            errors.append(f"Missing question IDs: {', '.join(missing)}")
        if unexpected:
            errors.append(f"Unexpected question IDs: {', '.join(unexpected)}")
        for question_id, value in submission.items():
            if not isinstance(question_id, str):
                continue
            if not isinstance(value, dict):
                errors.append(f"Answer for {question_id} must be an object.")
                continue
            if set(value) != {"answer"}:
                errors.append(
                    f"Answer object for {question_id} must contain only 'answer'."
                )
                continue
            answer = value["answer"]
            if not isinstance(answer, str):
                errors.append(f"answer for {question_id} must be a string.")
            elif not answer.strip():
                errors.append(f"answer for {question_id} must not be empty.")
        return SubmissionValidationResult(
            valid=not errors,
            errors=errors,
            missing_question_ids=missing,
            unexpected_question_ids=unexpected,
        )
