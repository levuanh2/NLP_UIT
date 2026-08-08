"""Stable hierarchy-aware parent-child legal chunking."""

import hashlib
import re
from collections.abc import Iterator

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument
from app.domain.metadata import LegalMetadata
from app.ingestion.chunking.base import BaseChunker


class ParentChildChunker(BaseChunker):
    def __init__(
        self,
        parent_target_tokens: int,
        parent_max_tokens: int,
        child_target_tokens: int,
        child_max_tokens: int,
    ) -> None:
        values = (
            parent_target_tokens,
            parent_max_tokens,
            child_target_tokens,
            child_max_tokens,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Chunk token limits must be positive.")
        if parent_target_tokens > parent_max_tokens:
            raise ValueError("Parent target tokens cannot exceed parent maximum.")
        if child_target_tokens > child_max_tokens:
            raise ValueError("Child target tokens cannot exceed child maximum.")
        self.parent_target_tokens = parent_target_tokens
        self.parent_max_tokens = parent_max_tokens
        self.child_target_tokens = child_target_tokens
        self.child_max_tokens = child_max_tokens

    def chunk(
        self, document: LegalDocument
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        if document.structure is None:
            raise ValueError("Document structure must be extracted before chunking.")
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for chapter in document.structure.chapters:
            for section in chapter.sections:
                for article in section.articles:
                    header = f"Điều {article.article_number}"
                    if article.title:
                        header += f". {article.title}"
                    body = "\n".join(
                        clause.text for clause in article.clauses if clause.text
                    ).strip()
                    article_text = f"{header}\n{body}".strip()
                    parent_windows = list(
                        _windows(
                            article_text,
                            self.parent_target_tokens,
                            overlap=max(0, self.child_target_tokens // 2),
                        )
                    )
                    for parent_index, parent_text in enumerate(parent_windows):
                        parent_id = _stable_id(
                            document.document_id,
                            f"{article.article_id}:parent-{parent_index}",
                            parent_text,
                        )
                        parent = ParentChunk(
                            parent_id=parent_id,
                            document_id=document.document_id,
                            chapter=chapter.title,
                            section=section.title,
                            article=article.article_number,
                            text=parent_text,
                            token_count=len(parent_text.split()),
                        )
                        parents.append(parent)
                        for child_index, child_text in enumerate(
                            _windows(
                                parent_text,
                                self.child_target_tokens,
                                overlap=max(0, self.child_target_tokens // 5),
                            )
                        ):
                            clause, point = _labels(child_text)
                            child_id = _stable_id(
                                document.document_id,
                                f"{parent_id}:child-{child_index}",
                                child_text,
                            )
                            metadata = LegalMetadata(
                                document_id=document.document_id,
                                document_name=document.document_name,
                                source_link=document.source_link,
                                chapter=chapter.title,
                                section=section.title,
                                article=article.article_number,
                                clause=clause,
                                point=point,
                            )
                            children.append(
                                ChildChunk(
                                    child_id=child_id,
                                    parent_id=parent_id,
                                    document_id=document.document_id,
                                    chapter=chapter.title,
                                    section=section.title,
                                    article=article.article_number,
                                    clause=clause,
                                    point=point,
                                    original_text=child_text,
                                    embedding_text=(
                                        f"{document.document_name}\n{header}\n{child_text}"
                                    ),
                                    token_count=len(child_text.split()),
                                    metadata=metadata,
                                )
                            )
        return parents, children


def _windows(text: str, size: int, overlap: int) -> Iterator[str]:
    words = text.split()
    if not words:
        return
    step = max(1, size - min(overlap, size - 1))
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if window:
            yield " ".join(window)
        if start + size >= len(words):
            break


def _stable_id(document_id: int, scope: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"doc-{document_id}:{scope}:{digest}"


def _labels(text: str) -> tuple[str | None, str | None]:
    clause = None
    for match in re.finditer(r"(?m)(?:^|\s)(\d+)\.\s", text):
        prefix = text[max(0, match.start() - 6) : match.start()].casefold()
        if not prefix.rstrip().endswith("điều"):
            clause = match.group(1)
            break
    point_match = re.search(r"(?im)(?:^|\s)([a-zđ])\)\s", text)
    return (
        clause,
        point_match.group(1).casefold() if point_match else None,
    )
