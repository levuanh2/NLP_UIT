"""Streaming parent-child chunking for Vietnamese legal documents."""

import re
from collections.abc import Iterator
from itertools import chain

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalArticle, LegalDocument
from app.domain.metadata import LegalMetadata
from app.ingestion.chunking.base import BaseChunker


class ParentChildChunker(BaseChunker):
    """Use articles as parents and clauses/points as retrieval children."""

    def __init__(
        self,
        parent_target_tokens: int = 1024,
        parent_max_tokens: int = 1200,
        child_target_tokens: int = 256,
        child_max_tokens: int = 320,
        parent_min_tokens: int = 800,
        child_min_tokens: int = 220,
        child_overlap_tokens: int = 48,
    ) -> None:
        if not 0 <= child_overlap_tokens < child_target_tokens:
            raise ValueError("child overlap must be smaller than target size")
        self.parent_target_tokens = parent_target_tokens
        self.parent_min_tokens = parent_min_tokens
        self.parent_max_tokens = parent_max_tokens
        self.child_target_tokens = child_target_tokens
        self.child_min_tokens = child_min_tokens
        self.child_max_tokens = child_max_tokens
        self.child_overlap_tokens = child_overlap_tokens

    def chunk(
        self, document: LegalDocument
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Materialize chunks for small callers; ingestion uses the iterator."""
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for parent, group in self.iter_chunk_groups(document):
            parents.append(parent)
            children.extend(group)
        return parents, children

    def iter_chunk_groups(
        self, document: LegalDocument
    ) -> Iterator[tuple[ParentChunk, list[ChildChunk]]]:
        """Keep only the current parent and its child chunks in memory."""
        child_position = 0
        parent_position = 0
        article_iterator = self._iter_articles(document)
        first_article = next(article_iterator, None)
        if first_article is not None:
            for chapter, section, article in chain([first_article], article_iterator):
                article_label = f"Điều {article.article_number}"
                long_article = self._token_count(article.text) > self.parent_max_tokens
                parent_segments = self._iter_bounded_segments(
                    article.text, self.parent_target_tokens, overlap=0
                )
                for segment_index, parent_text in enumerate(parent_segments):
                    parent_id = (
                        f"doc:{document.document_id}:article:{article.position}:"
                        f"{article.article_number}:segment:{segment_index}"
                    )
                    method = "article_segment" if long_article else "article"
                    parent = self._parent(
                        document,
                        parent_id,
                        chapter,
                        section,
                        article_label,
                        parent_position,
                        parent_text,
                        method,
                    )
                    if long_article:
                        units = [
                            (None, None, text, "token_window")
                            for text in self._iter_bounded_segments(
                                parent_text,
                                self.child_target_tokens,
                                self.child_overlap_tokens,
                            )
                        ]
                    else:
                        units = self._structured_units(article)
                        if not units:
                            units = [
                                (None, None, text, "token_window")
                                for text in self._iter_bounded_segments(
                                    parent_text,
                                    self.child_target_tokens,
                                    self.child_overlap_tokens,
                                )
                            ]
                    children, child_position = self._children(
                        document,
                        parent,
                        units,
                        child_position,
                    )
                    yield parent, children
                    parent_position += 1
            return

        text = document.cleaned_text or document.raw_text
        for parent_text in self._iter_bounded_segments(
            text, self.parent_target_tokens, overlap=0
        ):
            parent_id = f"doc:{document.document_id}:fallback:{parent_position}"
            parent = self._parent(
                document,
                parent_id,
                None,
                None,
                None,
                parent_position,
                parent_text,
                "token_window",
            )
            units = [
                (None, None, child_text, "token_window")
                for child_text in self._iter_bounded_segments(
                    parent_text,
                    self.child_target_tokens,
                    self.child_overlap_tokens,
                )
            ]
            children, child_position = self._children(
                document, parent, units, child_position
            )
            yield parent, children
            parent_position += 1

    @staticmethod
    def _iter_articles(
        document: LegalDocument,
    ) -> Iterator[tuple[str | None, str | None, LegalArticle]]:
        structure = document.structure
        if structure is None:
            return
        for chapter in structure.chapters:
            for article in chapter.articles:
                yield chapter.title, None, article
            for section in chapter.sections:
                for article in section.articles:
                    yield chapter.title, section.title, article
        for section in structure.sections:
            for article in section.articles:
                yield None, section.title, article
        for article in structure.articles:
            yield None, None, article

    def _structured_units(
        self, article: LegalArticle
    ) -> list[tuple[str | None, str | None, str, str]]:
        units: list[tuple[str | None, str | None, str, str]] = []
        for clause in article.clauses:
            clause_label = f"Khoản {clause.clause_number}"
            if clause.points:
                for point in clause.points:
                    units.extend(
                        (
                            clause_label,
                            f"Điểm {point.point_label}",
                            segment,
                            "point" if index == 0 else "point_window",
                        )
                        for index, segment in enumerate(
                            self._iter_bounded_segments(
                                point.text,
                                self.child_target_tokens,
                                self.child_overlap_tokens,
                            )
                        )
                    )
            else:
                units.extend(
                    (
                        clause_label,
                        None,
                        segment,
                        "clause" if index == 0 else "clause_window",
                    )
                    for index, segment in enumerate(
                        self._iter_bounded_segments(
                            clause.text,
                            self.child_target_tokens,
                            self.child_overlap_tokens,
                        )
                    )
                )
        return units

    def _children(
        self,
        document: LegalDocument,
        parent: ParentChunk,
        units: list[tuple[str | None, str | None, str, str]],
        start_position: int,
    ) -> tuple[list[ChildChunk], int]:
        children: list[ChildChunk] = []
        for local_position, (clause, point, text, method) in enumerate(units):
            child_id = f"{parent.parent_id}:child:{local_position}"
            metadata = LegalMetadata(
                document_id=document.document_id,
                document_name=document.document_name,
                source_link=document.source_link,
                chapter=parent.chapter,
                section=parent.section,
                article=parent.article,
                clause=clause,
                point=point,
            )
            children.append(
                ChildChunk(
                    child_id=child_id,
                    parent_id=parent.parent_id,
                    document_id=document.document_id,
                    document_name=document.document_name,
                    source_link=document.source_link,
                    chapter=parent.chapter,
                    section=parent.section,
                    article=parent.article,
                    clause=clause,
                    point=point,
                    position=start_position + local_position,
                    text=text,
                    embedding_text=self._embedding_text(document, metadata, text),
                    token_count=self._token_count(text),
                    chunking_method=method,
                    metadata=metadata,
                )
            )
        for index, child in enumerate(children):
            child.previous_child_id = children[index - 1].child_id if index else None
            child.next_child_id = (
                children[index + 1].child_id if index + 1 < len(children) else None
            )
        return children, start_position + len(children)

    @staticmethod
    def _embedding_text(
        document: LegalDocument, metadata: LegalMetadata, text: str
    ) -> str:
        path = [
            document.document_name or f"Văn bản {document.document_id}",
            metadata.chapter,
            metadata.section,
            metadata.article,
            metadata.clause,
            metadata.point,
        ]
        return f"{' > '.join(item for item in path if item)}\n\n{text}"

    def _parent(
        self,
        document: LegalDocument,
        parent_id: str,
        chapter: str | None,
        section: str | None,
        article: str | None,
        position: int,
        text: str,
        method: str,
    ) -> ParentChunk:
        return ParentChunk(
            parent_id=parent_id,
            document_id=document.document_id,
            document_name=document.document_name,
            source_link=document.source_link,
            chapter=chapter,
            section=section,
            article=article,
            position=position,
            text=text,
            token_count=self._token_count(text),
            chunking_method=method,
        )

    @staticmethod
    def _token_count(text: str) -> int:
        return sum(1 for _ in re.finditer(r"\S+", text))

    @staticmethod
    def _iter_bounded_segments(text: str, size: int, overlap: int) -> Iterator[str]:
        """Split with a bounded token buffer; never tokenize a full document."""
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("invalid window size or overlap")
        buffer: list[str] = []
        emitted = False
        for match in re.finditer(r"\S+", text):
            buffer.append(match.group(0))
            if len(buffer) == size:
                yield " ".join(buffer)
                emitted = True
                buffer = buffer[-overlap:] if overlap else []
        if buffer and (not emitted or len(buffer) > overlap):
            yield " ".join(buffer)
