"""Versioned index manifest contract."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class IndexManifest(BaseModel):
    index_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    corpus_path: str
    document_count: int
    valid_document_count: int
    failed_document_count: int
    parent_count: int
    child_count: int
    embedding_model: str
    embedding_dimension: int
    chunking_version: str
    faiss_index_type: str
    status: str = "building"

    def validation_issues(self) -> list[str]:
        issues: list[str] = []
        if (
            self.document_count
            != self.valid_document_count + self.failed_document_count
        ):
            issues.append("document totals are inconsistent")
        if self.valid_document_count and not self.child_count:
            issues.append("valid documents exist but child count is zero")
        if self.child_count and self.embedding_dimension <= 0:
            issues.append("embedding dimension must be positive")
        if self.status not in {"building", "ready", "failed"}:
            issues.append(f"invalid manifest status: {self.status}")
        return issues
