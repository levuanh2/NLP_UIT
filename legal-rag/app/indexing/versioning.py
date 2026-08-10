"""Create and atomically publish local index versions."""

import os
import re
from pathlib import Path

from app.indexing.manifest import IndexManifest


class IndexVersionManager:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def create_version(self) -> str:
        """Create the next monotonic vN directory."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        numbers = [
            int(match.group(1))
            for path in self.root_dir.iterdir()
            if path.is_dir() and (match := re.fullmatch(r"v(\d+)", path.name))
        ]
        version = f"v{max(numbers, default=0) + 1}"
        self.ensure_version(version)
        return version

    def ensure_version(self, version: str) -> Path:
        """Create an explicit version and its bounded-index subdirectories."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
            raise ValueError(f"Invalid index version: {version}")
        version_dir = self.root_dir / version
        for child in ("faiss", "bm25", "metadata"):
            (version_dir / child).mkdir(parents=True, exist_ok=True)
        return version_dir

    def get_current_version(self) -> str | None:
        """Read the published CURRENT pointer."""
        current = self.root_dir / "CURRENT"
        if not current.is_file():
            return None
        value = current.read_text(encoding="utf-8").strip()
        return value or None

    def publish(self, version: str) -> None:
        """Atomically switch CURRENT only to a validated ready index."""
        version_dir = self.root_dir / version
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing manifest for index version {version}")
        manifest = IndexManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        issues = manifest.validation_issues()
        if manifest.status != "ready" or issues:
            detail = "; ".join(issues) if issues else manifest.status
            raise ValueError(f"Index version is not publishable: {detail}")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.root_dir / "CURRENT.tmp"
        temporary.write_text(f"{version}\n", encoding="utf-8")
        os.replace(temporary, self.root_dir / "CURRENT")

    def rollback(self, version: str) -> None:
        """Move CURRENT back to another previously ready version."""
        self.publish(version)

    def write_manifest(self, manifest: IndexManifest) -> Path:
        version_dir = self.ensure_version(manifest.index_version)
        path = version_dir / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return path
