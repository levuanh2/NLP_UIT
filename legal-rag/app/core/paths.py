"""Centralized, configurable filesystem paths."""

from pathlib import Path

from pydantic import BaseModel


class ProjectPaths(BaseModel):
    """Resolved paths used by the application."""

    project_root: Path
    data_dir: Path
    corpus_data_dir: Path
    question_data_dir: Path
    output_dir: Path
    cache_data_dir: Path
    model_dir: Path
    faiss_dir: Path
    bm25_dir: Path
    sqlite_dir: Path
    sqlite_database_path: Path
    index_root_dir: Path
    checkpoint_dir: Path
    config_dir: Path

    def ensure_runtime_directories(self) -> None:
        """Create configured runtime directories when explicitly called."""
        for path in (
            self.data_dir,
            self.corpus_data_dir,
            self.question_data_dir,
            self.output_dir,
            self.cache_data_dir,
            self.model_dir,
            self.faiss_dir,
            self.bm25_dir,
            self.sqlite_dir,
            self.index_root_dir,
            self.checkpoint_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
