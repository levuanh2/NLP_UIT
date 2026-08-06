"""Strict validation for the final submission JSON contract."""

from typing import Any

from app.domain.submission import SubmissionValidationResult


class SubmissionValidator:
    def validate(
        self,
        submission: dict[str, Any],
        expected_question_ids: set[str],
    ) -> SubmissionValidationResult:
        """Validate IDs and require each value to contain only a nonempty answer."""
        errors: list[str] = []
        if not isinstance(submission, dict):
            return SubmissionValidationResult(
                valid=False,
                errors=["Submission root must be a JSON object."],
                missing_question_ids=sorted(expected_question_ids),
                unexpected_question_ids=[],
            )

        actual_ids = set(submission)
        missing = sorted(expected_question_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_question_ids)
        if missing:
            errors.append(f"Missing question IDs: {', '.join(missing)}")
        if unexpected:
            errors.append(f"Unexpected question IDs: {', '.join(unexpected)}")

        for question_id, value in submission.items():
            if not isinstance(question_id, str):
                errors.append("Every question ID must be a string.")
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
