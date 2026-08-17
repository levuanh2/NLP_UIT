"""Time corpus embedding throughput on CPU vs CUDA.

Usage: python scripts/bench_embed.py [--device cpu|cuda] [--batch-size 64] [--texts 512]
"""

import argparse
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "storage/index-builds-qualified/v1/metadata/legal.sqlite"
REMAINING_CHILDREN = 610_000  # ~5730 unindexed docs at the observed 106 children/doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--texts", type=int, default=512)
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    texts = [
        row[0]
        for row in conn.execute(
            "SELECT embedding_text FROM child_chunks WHERE rowid % 7 = 0 LIMIT ?",
            (args.texts,),
        )
    ]
    conn.close()

    from app.indexing.embeddings.factory import EmbeddingModelFactory

    model = EmbeddingModelFactory.create(
        provider="sentence_transformers",
        model_name="bqbbao6/vietnamese-legal-embedding",
        device=args.device,
        local_files_only=True,
    )
    model.load()
    model.embed_documents(texts[: args.batch_size])  # warmup

    started = time.perf_counter()
    model.embed_documents(texts)
    elapsed = time.perf_counter() - started

    rate = len(texts) / elapsed
    print(
        f"device={args.device} batch={args.batch_size} texts={len(texts)}\n"
        f"  {elapsed:.1f}s -> {rate:.1f} children/s\n"
        f"  {REMAINING_CHILDREN:,} remaining children -> "
        f"{REMAINING_CHILDREN / rate / 3600:.1f} h of embedding alone"
    )
    model.unload()


if __name__ == "__main__":
    main()
