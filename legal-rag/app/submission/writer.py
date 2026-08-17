"""Validated atomic UTF-8 submission writer."""

import json
import os
from pathlib import Path

from app.domain.generation import GeneratedAnswer
from app.domain.submission import SubmissionAnswer
from app.submission.validator import SubmissionValidator


class SubmissionWriter:
    def __init__(self, encoding: str = "utf-8", ensure_ascii: bool = False) -> None:
        if encoding.lower().replace("-", "") != "utf8":
            raise ValueError("Competition submission encoding must be UTF-8")
        self.encoding = encoding
        # ensure_ascii escapes Vietnamese as \uXXXX rather than dropping it, so the
        # parsed answers are identical. A pure-ASCII file also survives a grader
        # that opens it under a non-UTF-8 locale, which raw Vietnamese does not.
        self.ensure_ascii = ensure_ascii

    def write(
        self,
        answers: dict[str, GeneratedAnswer | SubmissionAnswer],
        output_path: Path,
        *,
        expected_question_ids: set[str] | None = None,
    ) -> None:
        payload = {
            question_id: {
                "answer": answer.answer,
            }
            for question_id, answer in answers.items()
        }
        expected = expected_question_ids or set(answers)
        validation = SubmissionValidator().validate(payload, expected)
        if not validation.valid:
            raise ValueError("Invalid submission: " + "; ".join(validation.errors))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            # newline="\n": the organisers' grader reads the file, not Notepad.
            with temporary.open("w", encoding=self.encoding, newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=self.ensure_ascii, indent=4)
                stream.write("\n")
            disk_validation = SubmissionValidator().validate(temporary, expected)
            if not disk_validation.valid:
                raise ValueError(
                    "Written submission failed validation: "
                    + "; ".join(disk_validation.errors)
                )
            os.replace(temporary, output_path)
        finally:
            if temporary.exists():
                temporary.unlink()
