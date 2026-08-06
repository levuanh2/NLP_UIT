"""Strict JSON loading helpers used by submission commands."""

import json
from pathlib import Path
from typing import Any

from app.core.exceptions import SubmissionValidationError


def load_json_strict(path: Path) -> Any:
    """Load UTF-8 JSON while rejecting duplicate keys at every object level."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SubmissionValidationError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise SubmissionValidationError(
            f"JSON file is not valid UTF-8: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SubmissionValidationError(
            f"Invalid JSON syntax in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
