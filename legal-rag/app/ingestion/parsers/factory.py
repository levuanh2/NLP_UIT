"""Parser selection without parsing side effects."""

from pathlib import Path

from app.core.exceptions import UnsupportedDocumentError
from app.ingestion.parsers.base import BaseDocumentParser
from app.ingestion.parsers.json_context_parser import JsonContextParser


class DocumentParserFactory:
    def __init__(self, parsers: list[BaseDocumentParser] | None = None) -> None:
        self._parsers = parsers or [JsonContextParser()]

    def get_parser(self, source_path: Path) -> BaseDocumentParser:
        """Return the first parser supporting the source path."""
        for parser in self._parsers:
            if parser.supports(source_path):
                return parser
        raise UnsupportedDocumentError(f"Unsupported document: {source_path}")
