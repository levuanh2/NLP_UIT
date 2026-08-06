"""Convert generated answers to the exact Subtask 2 schema."""

from typing import Any

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

    def format_internal_results(
        self, results: Any
    ) -> dict[str, SubmissionAnswer]:
        """Strip internal fields and retain only question ID and answer."""
        submission: dict[str, SubmissionAnswer] = {}

        if isinstance(results, dict):
            records = [
                {**value, "question_id": question_id}
                if isinstance(value, dict)
                else {"question_id": question_id, "answer": value}
                for question_id, value in results.items()
            ]
        elif isinstance(results, list):
            records = results
        else:
            raise ValueError("Internal results root must be an object or list.")

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Internal result at index {index} must be an object.")
            question_id = record.get("question_id")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(
                    f"Internal result at index {index} needs a nonempty "
                    "string question_id."
                )
            if question_id in submission:
                raise ValueError(f"Duplicate question_id: {question_id}")
            if "answer" not in record:
                raise ValueError(f"Internal result {question_id} is missing answer.")
            answer = record["answer"]
            if not isinstance(answer, str):
                raise ValueError(f"answer for {question_id} must be a string.")
            submission[question_id] = SubmissionAnswer(answer=answer)
        return submission
