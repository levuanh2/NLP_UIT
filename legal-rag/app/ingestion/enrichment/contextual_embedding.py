"""Embedding-text contextualization."""

from app.domain.chunks import ChildChunk


class ContextualEmbeddingTextBuilder:
    def build(self, chunk: ChildChunk) -> str:
        """Build metadata-aware passage text for embedding."""
        labels = [chunk.metadata.document_name]
        if chunk.chapter:
            labels.append(chunk.chapter)
        if chunk.section:
            labels.append(chunk.section)
        if chunk.article:
            labels.append(f"Điều {chunk.article}")
        if chunk.clause:
            labels.append(f"Khoản {chunk.clause}")
        if chunk.point:
            labels.append(f"Điểm {chunk.point}")
        return (
            " | ".join(value for value in labels if value) + "\n" + chunk.original_text
        )
