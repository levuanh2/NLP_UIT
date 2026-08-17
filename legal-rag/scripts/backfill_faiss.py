"""Embed child chunks that exist in SQLite but are missing from the FAISS shards.

A killed ingestion loses the in-memory partial shard, and resume skips those
documents because they are already checkpointed as complete. Without this
backfill the job's ``FAISS vector count differs from SQLite child count`` check
fails and the index is never published.

Usage: python scripts/backfill_faiss.py [--index-root ...] [--version v1] [--dry-run]
"""

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-root", type=Path, default=ROOT / "storage/index-builds-qualified"
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    version_dir = args.index_root / args.version
    faiss_dir = version_dir / "faiss"
    sqlite_path = version_dir / "metadata" / "legal.sqlite"

    indexed: set[str] = set()
    for mapping in sorted(faiss_dir.glob("shard_*.index.ids.json")):
        indexed.update(json.loads(mapping.read_text(encoding="utf-8")))

    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    rows = list(conn.execute("SELECT child_id, embedding_text FROM child_chunks"))
    conn.close()

    missing = [(cid, text) for cid, text in rows if cid not in indexed]
    print(
        f"sqlite children={len(rows):,} faiss vectors={len(indexed):,} "
        f"missing={len(missing):,}"
    )
    if not missing:
        print("nothing to backfill")
        return
    if args.dry_run:
        return

    from app.indexing.embeddings.factory import EmbeddingModelFactory
    from app.indexing.vector_store.writer import FAISSShardWriter

    model = EmbeddingModelFactory.create(
        provider="sentence_transformers",
        model_name="bqbbao6/vietnamese-legal-embedding",
        device="cuda",
        local_files_only=True,
    )
    model.load()
    writer = FAISSShardWriter(faiss_dir, model.dimension())

    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        vectors = model.embed_documents([text for _, text in batch])
        writer.add_batch([cid for cid, _ in batch], vectors)
        done = start + len(batch)
        if done % 10_000 < args.batch_size or done == len(missing):
            print(f"  {done:,}/{len(missing):,}")
    writer.finalize()
    model.unload()

    total = sum(
        len(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(faiss_dir.glob("shard_*.index.ids.json"))
    )
    print(f"faiss vectors now {total:,} (sqlite children {len(rows):,})")
    assert total == len(rows), "backfill did not close the FAISS/SQLite gap"


if __name__ == "__main__":
    main()
