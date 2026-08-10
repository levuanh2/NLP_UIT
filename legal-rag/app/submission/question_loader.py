"""Non-mutating adapters for competition question datasets."""

import csv
import json
from pathlib import Path
from typing import Any

from app.domain.queries import LegalQuery


class QuestionDatasetLoader:
    def load(self, path: Path) -> list[LegalQuery]:
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open(encoding="utf-8-sig") as stream:
                return self._from_json(json.load(stream))
        if suffix == ".jsonl":
            with path.open(encoding="utf-8-sig") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            return self._records(records)
        if suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                return self._records(list(csv.DictReader(stream)))
        raise ValueError(f"Unsupported question dataset format: {suffix}")

    def _from_json(self, payload: Any) -> list[LegalQuery]:
        if isinstance(payload, list):
            return self._records(payload)
        if not isinstance(payload, dict):
            raise ValueError("Question JSON root must be an object or list")
        if "question_id" in payload and "question" in payload:
            return self._records([payload])
        records: list[dict[str, Any]] = []
        for question_id, value in payload.items():
            if isinstance(value, str):
                records.append({"question_id": question_id, "question": value})
            elif isinstance(value, dict):
                records.append(
                    {
                        "question_id": question_id,
                        "question": value.get("question"),
                    }
                )
            else:
                raise ValueError(f"Invalid question record for ID {question_id}")
        return self._records(records)

    @staticmethod
    def _records(records: list[Any]) -> list[LegalQuery]:
        questions: list[LegalQuery] = []
        seen: set[str] = set()
        for position, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Question record {position} must be an object")
            question_id = str(record.get("question_id", "")).strip()
            question = record.get("question")
            if not question_id or not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Question record {position} needs non-empty question_id/question"
                )
            if question_id in seen:
                raise ValueError(f"Duplicate question_id: {question_id}")
            seen.add(question_id)
            questions.append(
                LegalQuery(question_id=question_id, question=question.strip())
            )
        return questions
