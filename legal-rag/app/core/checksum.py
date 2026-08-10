"""Streaming checksums used by idempotent ingestion."""

import hashlib
from pathlib import Path


def calculate_file_checksum(path: Path, block_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without reading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
