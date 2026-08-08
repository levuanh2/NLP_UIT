"""Competition question and answer dataset loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.submission.json_io import load_json_strict


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    """One question with an optional reference answer."""

    question_id: str
    question: str
    answer: str | None = None


def load_question_records(path: Path, require_answers: bool) -> list[QuestionRecord]:
    """Load the official ID-keyed format or a list of question records."""
    payload = load_json_strict(path)
    records: list[QuestionRecord] = []

    if isinstance(payload, dict):
        items: list[tuple[Any, Any]] = list(payload.items())
    elif isinstance(payload, list):
        items = [
            (item.get("question_id"), item) if isinstance(item, dict) else (None, item)
            for item in payload
        ]
    else:
        raise ValueError("Question dataset root must be an object or list.")

    seen_ids: set[str] = set()
    for index, (raw_id, value) in enumerate(items):
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"Question at index {index} has an invalid ID.")
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate question_id: {raw_id}")
        seen_ids.add(raw_id)
        if not isinstance(value, dict):
            raise ValueError(f"Question {raw_id} must be an object.")
        question = value.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {raw_id} has no nonempty question text.")
        answer = value.get("answer")
        if require_answers and (not isinstance(answer, str) or not answer.strip()):
            raise ValueError(f"Training question {raw_id} has no usable answer.")
        if answer is not None and not isinstance(answer, str):
            raise ValueError(f"answer for {raw_id} must be a string or null.")
        records.append(QuestionRecord(raw_id, question.strip(), answer))
    return records
