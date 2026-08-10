"""SQLAlchemy database construction without import-time connections."""

from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


class Database:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    def initialize(self) -> None:
        """Initialize the local SQLite engine when explicitly requested."""
        from app.indexing.metadata_store.models import Base

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self.database_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)

    def session(self) -> Session:
        if self._session_factory is None:
            raise RuntimeError("Database has not been initialized")
        return self._session_factory()
