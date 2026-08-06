"""Convert generated answers to the exact Subtask 2 schema."""

from app.domain.generation import GeneratedAnswer
from app.domain.submission import SubmissionAnswer


class SubmissionFormatter:
    def format(
        self, answers: list[GeneratedAnswer]
    ) -> dict[str, SubmissionAnswer]:
        """Use question IDs as keys and retain only the answer field."""
        submission: dict[str, SubmissionAnswer] = {}
        for generated in answers:
            if generated.question_id in submission:
                raise ValueError(
                    f"Duplicate question_id: {generated.question_id}"
                )
            submission[generated.question_id] = SubmissionAnswer(
                answer=generated.answer
            )
        return submission
