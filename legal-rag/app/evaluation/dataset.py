"""Evaluation dataset loading."""

from pathlib import Path

from app.baseline.data import load_question_records
from app.domain.evaluation import EvaluationSample
from app.submission.json_io import load_json_strict


class EvaluationDatasetLoader:
    def load(self, path: Path) -> list[EvaluationSample]:
        payload = load_json_strict(path)
        records = load_question_records(path, require_answers=False)
        samples: list[EvaluationSample] = []
        for record in records:
            raw = (
                payload.get(record.question_id, {}) if isinstance(payload, dict) else {}
            )
            relevant = (
                raw.get("relevant_child_ids", []) if isinstance(raw, dict) else []
            )
            if not isinstance(relevant, list) or not all(
                isinstance(item, str) for item in relevant
            ):
                raise ValueError(f"Invalid relevant_child_ids for {record.question_id}")
            samples.append(
                EvaluationSample(
                    question_id=record.question_id,
                    question=record.question,
                    expected_answer=record.answer,
                    relevant_child_ids=relevant,
                )
            )
        return samples
