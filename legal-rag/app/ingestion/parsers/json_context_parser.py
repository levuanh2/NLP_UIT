"""Parser for one competition ``context_*.json`` file."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.exceptions import DocumentParseError
from app.domain.documents import LegalDocument
from app.ingestion.parsers.base import BaseDocumentParser


class JsonContextRecord(BaseModel):
    """Validated wire schema of one competition context file."""

    model_config = ConfigDict(strict=True, extra="ignore")

    id: int
    name: str | None = Field(default=None, min_length=1)
    link: str = Field(min_length=1)
    passage: str = Field(min_length=1)


class JsonContextParser(BaseDocumentParser):
    def supports(self, source_path: Path) -> bool:
        """Return ``True`` only for JSON files."""
        return source_path.suffix.lower() == ".json"

    def parse(self, source_path: Path) -> LegalDocument:
        """Read and map one competition context JSON file."""
        if not self.supports(source_path):
            raise DocumentParseError(f"Expected a JSON context file: {source_path}")
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            record = JsonContextRecord.model_validate(payload)
        except OSError as exc:
            raise DocumentParseError(
                f"Cannot read context file: {source_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise DocumentParseError(
                f"Invalid JSON in context file: {source_path}"
            ) from exc
        except ValidationError as exc:
            raise DocumentParseError(
                f"Invalid context schema in {source_path}: {exc}"
            ) from exc

        return LegalDocument(
            document_id=record.id,
            document_name=record.name or "",
            source_link=record.link,
            raw_text=record.passage,
        )
