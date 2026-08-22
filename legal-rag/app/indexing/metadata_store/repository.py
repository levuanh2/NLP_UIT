"""Transactional SQLite repository for documents and hierarchical chunks."""

from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import aliased

from app.domain import legal_identifier
from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument
from app.domain.metadata import LegalMetadata
from app.domain.queries import QueryMetadata
from app.indexing.metadata_store.database import Database
from app.indexing.metadata_store.models import (
    ChildChunkRecord,
    DocumentRecord,
    IngestionErrorRecord,
    IngestionJobRecord,
    ParentChunkRecord,
)


class LegalRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save_document(
        self,
        document: LegalDocument,
        checksum: str,
        chunking_version: str,
        index_version: str,
        status: str,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ) -> None:
        with self.database.session() as session, session.begin():
            existing = session.get(DocumentRecord, document.document_id)
            created_at = existing.created_at if existing else datetime.utcnow()
            session.merge(
                DocumentRecord(
                    document_id=document.document_id,
                    document_name=document.document_name,
                    source_link=document.source_link,
                    checksum=checksum,
                    has_passage=bool(document.raw_text.strip()),
                    chunking_version=chunking_version,
                    embedding_model=embedding_model,
                    embedding_dimension=embedding_dimension,
                    index_version=index_version,
                    status=status,
                    created_at=created_at,
                    updated_at=datetime.utcnow(),
                )
            )

    def is_document_current(
        self,
        document_id: int,
        checksum: str,
        chunking_version: str,
        index_version: str,
    ) -> bool:
        with self.database.session() as session:
            record = session.get(DocumentRecord, document_id)
            return bool(
                record
                and record.checksum == checksum
                and record.chunking_version == chunking_version
                and record.index_version == index_version
                and record.status == "completed"
            )

    def delete_document_chunks(self, document_id: int) -> None:
        """Clear partial/stale rows before an idempotent document rebuild."""
        with self.database.session() as session, session.begin():
            session.execute(
                delete(ChildChunkRecord).where(
                    ChildChunkRecord.document_id == document_id
                )
            )
            session.execute(
                delete(ParentChunkRecord).where(
                    ParentChunkRecord.document_id == document_id
                )
            )

    def save_chunks(
        self, parent_chunks: list[ParentChunk], child_chunks: list[ChildChunk]
    ) -> None:
        """Persist one bounded micro-batch transactionally."""
        with self.database.session() as session, session.begin():
            if parent_chunks:
                parent_values = [parent.model_dump() for parent in parent_chunks]
                parent_insert = sqlite_insert(ParentChunkRecord)
                session.execute(
                    parent_insert.on_conflict_do_update(
                        index_elements=[ParentChunkRecord.parent_id],
                        set_={
                            column.name: getattr(parent_insert.excluded, column.name)
                            for column in ParentChunkRecord.__table__.columns
                            if column.name != "parent_id"
                        },
                    ),
                    parent_values,
                )
            if child_chunks:
                child_values = [
                    child.model_dump(exclude={"metadata"}) for child in child_chunks
                ]
                child_insert = sqlite_insert(ChildChunkRecord)
                session.execute(
                    child_insert.on_conflict_do_update(
                        index_elements=[ChildChunkRecord.child_id],
                        set_={
                            column.name: getattr(child_insert.excluded, column.name)
                            for column in ChildChunkRecord.__table__.columns
                            if column.name != "child_id"
                        },
                    ),
                    child_values,
                )

    def get_child(self, child_id: str) -> ChildChunk | None:
        with self.database.session() as session:
            record = session.get(ChildChunkRecord, child_id)
            return self._child(record) if record else None

    def get_parent(self, parent_id: str) -> ParentChunk | None:
        with self.database.session() as session:
            record = session.get(ParentChunkRecord, parent_id)
            return self._parent(record) if record else None

    def get_children_for_parent(self, parent_id: str) -> list[ChildChunk]:
        with self.database.session() as session:
            records = session.scalars(
                select(ChildChunkRecord)
                .where(ChildChunkRecord.parent_id == parent_id)
                .order_by(ChildChunkRecord.position)
            ).all()
            return [self._child(record) for record in records]

    def get_neighbor_children(self, child_id: str, window: int = 1) -> list[ChildChunk]:
        child = self.get_child(child_id)
        if child is None:
            return []
        with self.database.session() as session:
            records = session.scalars(
                select(ChildChunkRecord)
                .where(
                    ChildChunkRecord.parent_id == child.parent_id,
                    ChildChunkRecord.position.between(
                        child.position - window, child.position + window
                    ),
                )
                .order_by(ChildChunkRecord.position)
            ).all()
            return [self._child(record) for record in records]

    def document_ids_for_identifier(self, identifier: str) -> set[int]:
        """Documents whose slug carries this canonical legal identifier.

        Stored names are URL slugs ("Thong-tu-17-2022-TT-BGTVT-...-522401"), so
        the identifier is folded to its slug form and anchored on the leading
        hyphen: without the anchor "17-2022-TT-BGTVT" also matches "117-2022-...".
        """
        fragment = legal_identifier.escape_like(
            legal_identifier.slug_fragment(identifier)
        )
        with self.database.session() as session:
            return set(
                session.scalars(
                    select(ChildChunkRecord.document_id)
                    .where(
                        ChildChunkRecord.document_name.ilike(
                            f"%-{fragment}%", escape="\\"
                        )
                    )
                    .distinct()
                ).all()
            )

    def filter_child_ids(self, metadata: QueryMetadata) -> set[str]:
        conditions = []
        if metadata.document_id is not None:
            conditions.append(ChildChunkRecord.document_id == metadata.document_id)
        if metadata.document_number:
            documents = self.document_ids_for_identifier(metadata.document_number)
            if not documents:
                return set()
            conditions.append(ChildChunkRecord.document_id.in_(documents))
        if metadata.chapter:
            conditions.append(ChildChunkRecord.chapter == metadata.chapter)
        if metadata.section:
            conditions.append(ChildChunkRecord.section == metadata.section)
        if metadata.article:
            conditions.append(ChildChunkRecord.article == metadata.article)
        if metadata.clause:
            conditions.append(ChildChunkRecord.clause == metadata.clause)
        if metadata.point:
            conditions.append(ChildChunkRecord.point == metadata.point)
        if not conditions:
            return set()
        with self.database.session() as session:
            return set(
                session.scalars(
                    select(ChildChunkRecord.child_id).where(*conditions)
                ).all()
            )

    def get_metadata(self, child_id: str) -> LegalMetadata | None:
        child = self.get_child(child_id)
        return child.metadata if child else None

    def record_error(
        self,
        job_id: str,
        source_path: str,
        error_type: str,
        error_message: str,
        document_id: int | None = None,
    ) -> None:
        with self.database.session() as session, session.begin():
            session.add(
                IngestionErrorRecord(
                    job_id=job_id,
                    document_id=document_id,
                    source_path=source_path,
                    error_type=error_type,
                    error_message=error_message,
                )
            )

    def save_job(self, job: IngestionJobRecord) -> None:
        with self.database.session() as session, session.begin():
            session.merge(job)

    def counts(self) -> tuple[int, int, int]:
        with self.database.session() as session:
            documents = session.scalar(select(func.count()).select_from(DocumentRecord))
            parents = session.scalar(
                select(func.count()).select_from(ParentChunkRecord)
            )
            children = session.scalar(
                select(func.count()).select_from(ChildChunkRecord)
            )
            return int(documents or 0), int(parents or 0), int(children or 0)

    def validate_chunk_integrity(self) -> list[str]:
        issues: list[str] = []
        previous = aliased(ChildChunkRecord)
        following = aliased(ChildChunkRecord)
        with self.database.session() as session:
            missing_parents = session.scalar(
                select(func.count())
                .select_from(ChildChunkRecord)
                .outerjoin(
                    ParentChunkRecord,
                    ChildChunkRecord.parent_id == ParentChunkRecord.parent_id,
                )
                .where(ParentChunkRecord.parent_id.is_(None))
            )
            broken_previous = session.scalar(
                select(func.count())
                .select_from(ChildChunkRecord)
                .outerjoin(
                    previous, ChildChunkRecord.previous_child_id == previous.child_id
                )
                .where(
                    ChildChunkRecord.previous_child_id.is_not(None),
                    previous.child_id.is_(None),
                )
            )
            broken_next = session.scalar(
                select(func.count())
                .select_from(ChildChunkRecord)
                .outerjoin(
                    following, ChildChunkRecord.next_child_id == following.child_id
                )
                .where(
                    ChildChunkRecord.next_child_id.is_not(None),
                    following.child_id.is_(None),
                )
            )
            cross_parent_previous = session.scalar(
                select(func.count())
                .select_from(ChildChunkRecord)
                .join(previous, ChildChunkRecord.previous_child_id == previous.child_id)
                .where(previous.parent_id != ChildChunkRecord.parent_id)
            )
            cross_parent_next = session.scalar(
                select(func.count())
                .select_from(ChildChunkRecord)
                .join(following, ChildChunkRecord.next_child_id == following.child_id)
                .where(following.parent_id != ChildChunkRecord.parent_id)
            )
        if missing_parents:
            issues.append(f"missing parent references: {missing_parents}")
        if broken_previous:
            issues.append(f"broken previous-child references: {broken_previous}")
        if broken_next:
            issues.append(f"broken next-child references: {broken_next}")
        if cross_parent_previous:
            issues.append(f"cross-parent previous references: {cross_parent_previous}")
        if cross_parent_next:
            issues.append(f"cross-parent next references: {cross_parent_next}")
        return issues

    @staticmethod
    def _parent(record: ParentChunkRecord) -> ParentChunk:
        return ParentChunk(
            parent_id=record.parent_id,
            document_id=record.document_id,
            document_name=record.document_name,
            source_link=record.source_link,
            chapter=record.chapter,
            section=record.section,
            article=record.article,
            position=record.position,
            text=record.text,
            token_count=record.token_count,
            chunking_method=record.chunking_method,
        )

    @staticmethod
    def _child(record: ChildChunkRecord) -> ChildChunk:
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
            document_name=record.document_name,
            source_link=record.source_link,
            chapter=record.chapter,
            section=record.section,
            article=record.article,
            clause=record.clause,
            point=record.point,
            position=record.position,
            previous_child_id=record.previous_child_id,
            next_child_id=record.next_child_id,
            text=record.text,
            embedding_text=record.embedding_text,
            token_count=record.token_count,
            chunking_method=record.chunking_method,
            metadata=metadata,
        )
