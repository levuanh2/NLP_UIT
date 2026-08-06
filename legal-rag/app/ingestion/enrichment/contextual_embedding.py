"""Embedding-text contextualization skeleton."""

from app.domain.chunks import ChildChunk


class ContextualEmbeddingTextBuilder:
    def build(self, chunk: ChildChunk) -> str:
        """Build metadata-aware passage text for embedding."""
        # TODO(phase-implementation):
        # Prefix original text with concise legal hierarchy context.
        raise NotImplementedError
