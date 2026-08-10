"""Question dataset adapter tests."""

import json
from pathlib import Path

from app.submission.question_loader import QuestionDatasetLoader


def test_question_loader_competition_object(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {"80189": {"question": "Mẫu thông báo là gì?", "answer": None}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    questions = QuestionDatasetLoader().load(path)

    assert questions[0].question_id == "80189"
    assert questions[0].question == "Mẫu thông báo là gì?"
