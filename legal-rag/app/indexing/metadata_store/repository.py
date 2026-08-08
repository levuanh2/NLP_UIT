"""Transactional SQLite repository for legal chunks and metadata."""

from sqlalchemy import delete, func, insert, or_, select

from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.metadata import LegalMetadata
from app.domain.queries import QueryMetadata
from app.indexing.metadata_store.database import Database
from app.indexing.metadata_store.models import ChildChunkRecord, ParentChunkRecord


class LegalRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save_chunks(
        self, parent_chunks: list[ParentChunk], child_chunks: list[ChildChunk]
    ) -> None:
        parent_ids = {item.parent_id for item in parent_chunks}
        if len(parent_ids) != len(parent_chunks):
            raise ValueError("Duplicate parent chunk IDs.")
        child_ids = {item.child_id for item in child_chunks}
        if len(child_ids) != len(child_chunks):
            raise ValueError("Duplicate child chunk IDs.")
        if any(child.parent_id not in parent_ids for child in child_chunks):
            raise ValueError("Every child must reference a persisted parent.")
        documents = {
            item.document_id: (
                item.metadata.document_name,
                item.metadata.source_link,
            )
            for item in child_chunks
        }
        self.reset()
        self.add_parents(parent_chunks, documents)
        self.add_children(child_chunks)

    def reset(self) -> None:
        with self.database.session() as session, session.begin():
            session.execute(delete(ChildChunkRecord))
            session.execute(delete(ParentChunkRecord))

    def add_parents(
        self,
        parents: list[ParentChunk],
        documents: dict[int, tuple[str, str]] | None = None,
    ) -> None:
        if not parents:
            return
        documents = documents or {}
        rows = [
            {
                "parent_id": item.parent_id,
                "document_id": item.document_id,
                "document_name": documents.get(item.document_id, ("", ""))[0],
                "source_link": documents.get(item.document_id, ("", ""))[1],
                "chapter": item.chapter,
                "section": item.section,
                "article": item.article,
                "clause": None,
                "point": None,
                "text": item.text,
            }
            for item in parents
        ]
        with self.database.session() as session, session.begin():
            session.execute(insert(ParentChunkRecord), rows)

    def add_children(self, children: list[ChildChunk]) -> None:
        if not children:
            return
        rows = [
            {
                "child_id": item.child_id,
                "parent_id": item.parent_id,
                "document_id": item.document_id,
                "document_name": item.metadata.document_name,
                "source_link": item.metadata.source_link,
                "chapter": item.chapter,
                "section": item.section,
                "article": item.article,
                "clause": item.clause,
                "point": item.point,
                "original_text": item.original_text,
                "embedding_text": item.embedding_text,
            }
            for item in children
        ]
        with self.database.session() as session, session.begin():
            session.execute(insert(ChildChunkRecord), rows)

    def get_child(self, child_id: str) -> ChildChunk | None:
        with self.database.session() as session:
            record = session.get(ChildChunkRecord, child_id)
            return _map_child(record) if record is not None else None

    def get_children(self, child_ids: list[str]) -> dict[str, ChildChunk]:
        """Load a small ranked candidate set in one database round trip."""
        if not child_ids:
            return {}
        with self.database.session() as session:
            records = session.scalars(
                select(ChildChunkRecord).where(ChildChunkRecord.child_id.in_(child_ids))
            ).all()
            return {record.child_id: _map_child(record) for record in records}

    def get_parent(self, parent_id: str) -> ParentChunk | None:
        with self.database.session() as session:
            record = session.get(ParentChunkRecord, parent_id)
            if record is None:
                return None
            return ParentChunk(
                parent_id=record.parent_id,
                document_id=record.document_id,
                chapter=record.chapter,
                section=record.section,
                article=record.article,
                text=record.text,
                token_count=len(record.text.split()),
            )

    def filter_child_ids(self, metadata: QueryMetadata) -> set[str]:
        predicates = []
        if metadata.document_name:
            predicates.append(
                func.lower(ChildChunkRecord.document_name).contains(
                    metadata.document_name.casefold()
                )
            )
        if metadata.document_number:
            value = metadata.document_number.casefold()
            predicates.append(
                or_(
                    func.lower(ChildChunkRecord.document_name).contains(value),
                    func.lower(ChildChunkRecord.original_text).contains(value),
                )
            )
        if metadata.document_type:
            predicates.append(
                func.lower(ChildChunkRecord.document_name).contains(
                    metadata.document_type.casefold()
                )
            )
        if metadata.issued_year:
            predicates.append(
                ChildChunkRecord.document_name.contains(str(metadata.issued_year))
            )
        if metadata.article:
            predicates.append(ChildChunkRecord.article == metadata.article)
        if metadata.clause:
            predicates.append(ChildChunkRecord.clause == metadata.clause)
        if not predicates:
            return set()
        with self.database.session() as session:
            return set(
                session.scalars(
                    select(ChildChunkRecord.child_id).where(*predicates)
                ).all()
            )

    def get_metadata(self, child_id: str) -> LegalMetadata | None:
        child = self.get_child(child_id)
        return child.metadata if child is not None else None


def _child_record(item: ChildChunk) -> ChildChunkRecord:
    return ChildChunkRecord(
        child_id=item.child_id,
        parent_id=item.parent_id,
        document_id=item.document_id,
        document_name=item.metadata.document_name,
        source_link=item.metadata.source_link,
        chapter=item.chapter,
        section=item.section,
        article=item.article,
        clause=item.clause,
        point=item.point,
        original_text=item.original_text,
        embedding_text=item.embedding_text,
    )


def _map_child(record: ChildChunkRecord) -> ChildChunk:
    metadata = LegalMetadata(
        document_id=record.document_id,
        document_name=record.document_name,
        source_link=record.source_link,
        chapter=record.chapter,
        section=record.section,
        article=record.article,
        clause=record.clause,
        point=record.point,
    )
    return ChildChunk(
        child_id=record.child_id,
        parent_id=record.parent_id,
        document_id=record.document_id,
        chapter=record.chapter,
        section=record.section,
        article=record.article,
        clause=record.clause,
        point=record.point,
        original_text=record.original_text,
        embedding_text=record.embedding_text,
        token_count=len(record.original_text.split()),
        metadata=metadata,
    )
