"""Strict submission JSON loading tests."""

from pathlib import Path

import pytest

from app.core.exceptions import SubmissionValidationError
from app.submission.json_io import load_json_strict


def test_strict_loader_rejects_duplicate_question_key(tmp_path: Path) -> None:
    path = tmp_path / "submission.json"
    path.write_text(
        '{"1": {"answer": "A"}, "1": {"answer": "B"}}',
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="Duplicate JSON key: 1"):
        load_json_strict(path)


def test_strict_loader_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "submission.json"
    path.write_text('{"1": {"answer": "A"},}', encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="Invalid JSON syntax"):
        load_json_strict(path)
