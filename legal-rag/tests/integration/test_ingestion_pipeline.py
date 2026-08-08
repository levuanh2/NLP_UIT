"""Ingestion integration tests."""

import json
from pathlib import Path

from app.ingestion.chunking.parent_child import ParentChildChunker
from app.ingestion.cleaners.legal_text_cleaner import LegalTextCleaner
from app.ingestion.enrichment.metadata_enricher import MetadataEnricher
from app.ingestion.parsers.factory import DocumentParserFactory
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.structure.extractor import LegalStructureExtractor


def test_ingestion_pipeline_preserves_legal_hierarchy(tmp_path: Path) -> None:
    (tmp_path / "context_1.json").write_text(
        json.dumps(
            {
                "id": 1,
                "name": "Luật mẫu",
                "link": "https://example.test",
                "passage": "CHƯƠNG I\nĐIỀU 1. Phạm vi\n1. Nội dung.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pipeline = IngestionPipeline(
        DocumentParserFactory(),
        LegalTextCleaner(),
        LegalStructureExtractor(),
        ParentChildChunker(100, 120, 40, 60),
        MetadataEnricher(),
    )
    result = pipeline.run(tmp_path)
    assert len(result) == 1
    assert result[0].document.structure is not None
    assert result[0].parent_chunks and result[0].child_chunks
    assert result[0].child_chunks[-1].metadata.article == "1"
