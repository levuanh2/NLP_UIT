"""Competition dataset loader tests."""

import json
from pathlib import Path

import pytest

from app.baseline.data import load_question_records


def test_loads_official_id_keyed_dataset(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {"147194": {"question": "Quy định gì?", "answer": None}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = load_question_records(path, require_answers=False)

    assert records[0].question_id == "147194"
    assert records[0].question == "Quy định gì?"


def test_training_loader_requires_nonempty_answer(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text(
        '{"1": {"question": "Câu hỏi?", "answer": null}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no usable answer"):
        load_question_records(path, require_answers=True)
