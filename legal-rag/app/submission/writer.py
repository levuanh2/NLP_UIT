"""UTF-8 JSON submission writer."""

import json
from pathlib import Path

from app.core.exceptions import SubmissionValidationError
from app.domain.submission import SubmissionAnswer


class SubmissionWriter:
    def __init__(self, encoding: str = "utf-8", ensure_ascii: bool = False) -> None:
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii

    def write(
        self,
        submission: dict[str, SubmissionAnswer],
        output_path: Path,
    ) -> None:
        """Write the exact submission schema as indented UTF-8 JSON."""
        if output_path.name != "submission.json":
            raise SubmissionValidationError(
                "Subtask 2 output file must be named submission.json."
            )
        if self.encoding.lower().replace("_", "-") != "utf-8":
            raise SubmissionValidationError("Submission encoding must be UTF-8.")
        if self.ensure_ascii:
            raise SubmissionValidationError(
                "ensure_ascii must be False to preserve Vietnamese characters."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            question_id: answer.model_dump()
            for question_id, answer in submission.items()
        }
        with output_path.open("w", encoding=self.encoding) as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=self.ensure_ascii,
                indent=4,
            )
            stream.write("\n")
