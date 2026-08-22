"""Stage an index whose BM25 half is built from embedding_text.

Only the lexical half changes, so FAISS and the SQLite metadata are shared with
the production version by symlink rather than copied: the vectors are identical
and a 6 GB duplicate buys nothing. Production CURRENT is never touched — the
staging tree gets its own CURRENT, and a benchmark points INDEX_ROOT_DIR at it.

Usage:
  python scripts/build_enriched_index.py
  INDEX_ROOT_DIR=./storage/index-staging python scripts/retrieval_benchmark.py ...
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

from app.core.config import get_settings
from app.indexing.lexical.bm25_index import BM25IndexWriter
from app.retrieval.active_index import ActiveIndex

ROOT = Path(__file__).resolve().parents[1]
BATCH = 5000


class _Chunk:
    """The two fields BM25IndexWriter reads, straight from SQLite."""

    __slots__ = ("child_id", "text", "embedding_text")

    def __init__(self, child_id: str, text: str, embedding_text: str | None) -> None:
        self.child_id = child_id
        self.text = text
        self.embedding_text = embedding_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--staging-root", type=Path, default=ROOT / "storage/index-staging"
    )
    parser.add_argument("--version", default="v1-enriched")
    args = parser.parse_args()

    settings = get_settings()
    source = ActiveIndex(settings.index_root_dir)
    print(f"source index: {source.version} at {source.version_dir}")
    print(f"  bm25   {source.bm25_path}")
    print(f"  sqlite {source.sqlite_path}")

    target_root: Path = args.staging_root
    target = target_root / args.version
    if target.exists():
        print(f"BLOCKED: {target} already exists; remove it or pick another version")
        return 2
    (target / "bm25").mkdir(parents=True)

    # FAISS and the metadata store are byte-identical to production, so they are
    # linked. Anything that writes to them would be a bug, and the benchmark's
    # own fingerprint check would catch it.
    (target / "faiss").symlink_to(source.faiss_dir.resolve(), target_is_directory=True)
    (target / "metadata").symlink_to(
        source.sqlite_path.parent.resolve(), target_is_directory=True
    )

    connection = sqlite3.connect(f"file:{source.sqlite_path}?mode=ro", uri=True)
    total = connection.execute("SELECT count(*) FROM child_chunks").fetchone()[0]
    print(f"children to index: {total}")

    writer = BM25IndexWriter(target / "bm25")
    started = time.perf_counter()
    seen: set[str] = set()
    missing_embedding_text = 0
    cursor = connection.execute(
        "SELECT child_id, text, embedding_text FROM child_chunks"
    )
    batch: list[_Chunk] = []
    written = 0
    while True:
        rows = cursor.fetchmany(BATCH)
        if not rows:
            break
        for child_id, text, embedding_text in rows:
            if child_id in seen:
                print(f"BLOCKED: duplicate child_id in source: {child_id}")
                return 2
            seen.add(child_id)
            if not embedding_text:
                missing_embedding_text += 1
            batch.append(_Chunk(child_id, text or "", embedding_text))
        writer.add_batch(batch)
        written += len(batch)
        batch = []
        if written % 100000 < BATCH:
            elapsed = time.perf_counter() - started
            print(f"  {written}/{total}  {elapsed:.0f}s", flush=True)
    writer.finalize()
    connection.close()

    indexed = writer.count
    print(f"indexed {indexed} rows in {time.perf_counter() - started:.0f}s")
    print(
        "chunks without embedding_text (fell back to text): "
        f"{missing_embedding_text}"
    )

    if indexed != total:
        print(f"BLOCKED: indexed {indexed} rows but source has {total}")
        return 2

    manifest = json.loads(source.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("child_count") != total:
        print(
            f"BLOCKED: manifest child_count {manifest.get('child_count')} "
            f"does not match {total} indexed"
        )
        return 2
    manifest["index_version"] = args.version
    manifest["derived_from"] = source.version
    manifest["bm25_text_field"] = "embedding_text"
    manifest["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target_root / "CURRENT").write_text(args.version, encoding="utf-8")

    # Prove the staged tree resolves the same way production does before anyone
    # benchmarks against it.
    staged = ActiveIndex(target_root)
    print(
        f"staged index READY: {staged.version} "
        f"({staged.manifest.child_count} children)"
    )
    print(f"  bm25   {staged.bm25_path}")
    print(f"  faiss  {staged.faiss_dir} -> {staged.faiss_dir.resolve()}")
    print(f"  sqlite {staged.sqlite_path}")
    print()
    print(f"benchmark it with: INDEX_ROOT_DIR={target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
