"""Portable JSONL cache between corpus ingestion and index construction."""

import json
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from app.domain.chunks import ChildChunk, ParentChunk
from app.ingestion.pipeline import IngestionResult


def write_chunk_cache(
    results: Iterable[IngestionResult], cache_directory: Path
) -> tuple[int, int, int]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    parent_path = cache_directory / "parent_chunks.jsonl"
    child_path = cache_directory / "child_chunks.jsonl"
    manifest_path = cache_directory / "manifest.json"
    parent_tmp = _temporary(cache_directory, ".parents.jsonl")
    child_tmp = _temporary(cache_directory, ".children.jsonl")
    documents = parents = children = 0
    try:
        with (
            parent_tmp.open("w", encoding="utf-8") as parent_stream,
            child_tmp.open("w", encoding="utf-8") as child_stream,
        ):
            for result in results:
                documents += 1
                for parent in result.parent_chunks:
                    parent_stream.write(parent.model_dump_json() + "\n")
                    parents += 1
                for child in result.child_chunks:
                    child_stream.write(child.model_dump_json() + "\n")
                    children += 1
        parent_tmp.replace(parent_path)
        child_tmp.replace(child_path)
        manifest = {
            "schema_version": 1,
            "document_count": documents,
            "parent_count": parents,
            "child_count": children,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        parent_tmp.unlink(missing_ok=True)
        child_tmp.unlink(missing_ok=True)
    return documents, parents, children


def load_chunk_cache(
    cache_directory: Path,
    max_children: int | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    parent_path = cache_directory / "parent_chunks.jsonl"
    child_path = cache_directory / "child_chunks.jsonl"
    if not parent_path.is_file() or not child_path.is_file():
        raise FileNotFoundError(f"Incomplete chunk cache: {cache_directory}")
    with child_path.open(encoding="utf-8") as stream:
        children: list[ChildChunk] = []
        for line in stream:
            if line.strip():
                children.append(ChildChunk.model_validate_json(line))
                if max_children is not None and len(children) >= max_children:
                    break
    allowed_parents = (
        {item.parent_id for item in children} if max_children is not None else None
    )
    with parent_path.open(encoding="utf-8") as stream:
        parents = []
        for line in stream:
            if not line.strip():
                continue
            parent = ParentChunk.model_validate_json(line)
            if allowed_parents is None or parent.parent_id in allowed_parents:
                parents.append(parent)
    return parents, children


def iter_parent_cache(cache_directory: Path) -> Iterator[ParentChunk]:
    path = cache_directory / "parent_chunks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Parent chunk cache does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield ParentChunk.model_validate_json(line)


def iter_child_cache(cache_directory: Path) -> Iterator[ChildChunk]:
    path = cache_directory / "child_chunks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Child chunk cache does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield ChildChunk.model_validate_json(line)


def _temporary(directory: Path, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=directory, suffix=suffix, delete=False
    ) as stream:
        return Path(stream.name)
