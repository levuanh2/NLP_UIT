"""Competition JSON context parser tests."""

import json
from pathlib import Path

import pytest

from app.core.exceptions import DocumentParseError
from app.ingestion.parsers.factory import DocumentParserFactory
from app.ingestion.parsers.json_context_parser import JsonContextParser


def _write_context(path: Path, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": 740,
        "name": "Quyet-dinh-5868-QD-BYT-2018",
        "link": "https://example.test/legal/740",
        "passage": "Điều 1. Phạm vi điều chỉnh.",
    }
    payload.update(overrides)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def test_json_context_parser_reads_context_file(tmp_path: Path) -> None:
    # Arrange
    source = tmp_path / "context_000001.json"
    _write_context(source)

    # Act
    document = JsonContextParser().parse(source)

    # Assert
    assert document.raw_text == "Điều 1. Phạm vi điều chỉnh."


def test_json_context_parser_validates_schema(tmp_path: Path) -> None:
    # Arrange
    source = tmp_path / "context_000001.json"
    _write_context(source, id="740")

    # Act / Assert
    with pytest.raises(DocumentParseError, match="Invalid context schema"):
        JsonContextParser().parse(source)


def test_json_context_parser_maps_fields(tmp_path: Path) -> None:
    # Arrange
    source = tmp_path / "context_000001.json"
    payload = _write_context(source)

    # Act
    document = JsonContextParser().parse(source)

    # Assert
    assert document.document_id == payload["id"]
    assert document.document_name == payload["name"]
    assert document.source_link == payload["link"]
    assert document.raw_text == payload["passage"]
    assert document.cleaned_text is None
    assert document.structure is None


def test_json_context_parser_allows_missing_optional_name(tmp_path: Path) -> None:
    source = tmp_path / "context_000001.json"
    source.write_text(
        json.dumps(
            {
                "id": 740,
                "link": "https://example.test/legal/740",
                "passage": "Nội dung văn bản.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    document = JsonContextParser().parse(source)

    assert document.document_name == ""


def test_json_context_parser_rejects_invalid_json(tmp_path: Path) -> None:
    # Arrange
    source = tmp_path / "context_000001.json"
    source.write_text('{"id": 740,', encoding="utf-8")

    # Act / Assert
    with pytest.raises(DocumentParseError, match="Invalid JSON"):
        JsonContextParser().parse(source)


def test_json_context_parser_rejects_missing_required_fields(
    tmp_path: Path,
) -> None:
    # Arrange
    source = tmp_path / "context_000001.json"
    source.write_text('{"id": 740, "name": "Văn bản"}', encoding="utf-8")

    # Act / Assert
    with pytest.raises(DocumentParseError, match="Invalid context schema"):
        JsonContextParser().parse(source)


def test_document_parser_factory_selects_json_parser() -> None:
    # Arrange
    factory = DocumentParserFactory()

    # Act
    parser = factory.get_parser(Path("context_000001.json"))

    # Assert
    assert isinstance(parser, JsonContextParser)
