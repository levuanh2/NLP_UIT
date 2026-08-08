"""Conservative overlapping word-window chunking for legal passages."""

from app.corpus.fts import chunk_text
from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument
from app.domain.metadata import LegalMetadata
from app.ingestion.chunking.base import BaseChunker


class WindowChunker(BaseChunker):
    """Split one legal document into one parent and many child word windows."""

    def __init__(
        self,
        target_words: int = 350,
        overlap_words: int = 60,
    ) -> None:
        if target_words <= 0 or not 0 <= overlap_words < target_words:
            raise ValueError("Invalid chunk size or overlap.")
        self.target_words = target_words
        self.overlap_words = overlap_words

    def chunk(
        self, document: LegalDocument
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        passage = (document.cleaned_text or document.raw_text).strip()
        if not passage:
            return [], []

        parent_id = f"{document.document_id}:parent"
        parent = ParentChunk(
            parent_id=parent_id,
            document_id=document.document_id,
            chapter=None,
            section=None,
            article=None,
            text=passage,
            token_count=len(passage.split()),
        )
        metadata_base = LegalMetadata(
            document_id=document.document_id,
            document_name=document.document_name,
            source_link=document.source_link,
            chapter=None,
            section=None,
            article=None,
            clause=None,
            point=None,
        )
        children: list[ChildChunk] = []
        for chunk_index, text in enumerate(
            chunk_text(passage, self.target_words, self.overlap_words)
        ):
            child_id = f"{document.document_id}:{chunk_index}"
            children.append(
                ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    document_id=document.document_id,
                    chapter=None,
                    section=None,
                    article=None,
                    clause=None,
                    point=None,
                    original_text=text,
                    embedding_text=text,
                    token_count=len(text.split()),
                    metadata=metadata_base.model_copy(),
                )
            )
        return [parent], children
