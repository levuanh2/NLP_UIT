"""Resolve and validate the atomically published retrieval index."""

from pathlib import Path

from app.indexing.manifest import IndexManifest
from app.indexing.versioning import IndexVersionManager


class ActiveIndex:
    """Validated paths for the version named by ``storage/indexes/CURRENT``."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        version = IndexVersionManager(root_dir).get_current_version()
        if version is None:
            raise RuntimeError(
                f"Active index pointer is missing: {root_dir / 'CURRENT'}"
            )
        self.version = version
        self.version_dir = root_dir / version
        self.manifest_path = self.version_dir / "manifest.json"
        if not self.manifest_path.is_file():
            raise RuntimeError(
                f"Active index manifest is missing: {self.manifest_path}"
            )
        self.manifest = IndexManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.index_version != version:
            raise RuntimeError(
                "CURRENT and manifest index versions differ: "
                f"{version!r} != {self.manifest.index_version!r}"
            )
        issues = self.manifest.validation_issues()
        if self.manifest.status != "ready" or issues:
            detail = "; ".join(issues) if issues else self.manifest.status
            raise RuntimeError(f"Active index is not READY: {detail}")

        self.faiss_dir = self.version_dir / "faiss"
        self.bm25_path = self.version_dir / "bm25" / "bm25.sqlite"
        self.sqlite_path = self.version_dir / "metadata" / "legal.sqlite"
        shards = sorted(self.faiss_dir.glob("shard_*.index"))
        if not shards:
            raise RuntimeError(f"Active FAISS index is missing: {self.faiss_dir}")
        for shard in shards:
            mapping = shard.with_suffix(shard.suffix + ".ids.json")
            if not mapping.is_file():
                raise RuntimeError(f"FAISS child-ID mapping is missing: {mapping}")
        if not self.bm25_path.is_file():
            raise RuntimeError(f"Active BM25 index is missing: {self.bm25_path}")
        if not self.sqlite_path.is_file():
            raise RuntimeError(f"Active SQLite metadata is missing: {self.sqlite_path}")
        self.faiss_shards = shards
