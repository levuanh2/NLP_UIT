"""Time the reranker on real child chunks, then extrapolate to the full question set.

Usage: python scripts/bench_rerank.py [--device cpu|cuda] [--queries 5]
"""

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "storage/index-builds-qualified/v1/metadata/legal.sqlite"
QUESTIONS = ROOT / "data/questions/public-official.json"
CANDIDATES = 30  # RRF_TOP_K
MAX_LENGTH = 2304
BATCH_SIZE = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    questions = list(json.loads(QUESTIONS.read_text(encoding="utf-8")).values())
    total_questions = len(questions)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    # ponytail: random-ish sample via rowid stride, no ORDER BY RANDOM() over 700MB
    texts = [
        row[0]
        for row in conn.execute(
            "SELECT text FROM child_chunks "
            "WHERE rowid % 997 = 0 AND token_count >= ? LIMIT ?",
            (args.min_tokens, CANDIDATES * args.queries),
        )
    ]
    conn.close()
    assert len(texts) >= CANDIDATES, f"only {len(texts)} chunks sampled"

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "AITeamVN/Vietnamese_Reranker"
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, local_files_only=True, dtype=getattr(torch, args.dtype)
    )
    model.to(args.device).eval()
    load_seconds = time.perf_counter() - load_start

    params = sum(p.numel() for p in model.parameters())
    lengths = [
        len(tokenizer(text, truncation=True, max_length=MAX_LENGTH)["input_ids"])
        for text in texts[:CANDIDATES]
    ]

    per_query: list[float] = []
    with torch.inference_mode():
        for i in range(args.queries):
            query = questions[i]["question"]
            batch_texts = texts[(i * CANDIDATES) % len(texts) :][:CANDIDATES]
            if len(batch_texts) < CANDIDATES:
                batch_texts = texts[:CANDIDATES]
            start = time.perf_counter()
            for s in range(0, len(batch_texts), args.batch_size):
                chunk = batch_texts[s : s + args.batch_size]
                encoded = tokenizer(
                    [query] * len(chunk),
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {k: v.to(args.device) for k, v in encoded.items()}
                model(**encoded)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            per_query.append(elapsed)
            print(f"  query {i + 1}: {elapsed:.2f}s")

    warm = per_query[1:] or per_query  # drop first (warmup)
    median = statistics.median(warm)
    total = median * total_questions
    print(
        f"\ndevice={args.device} dtype={args.dtype} batch={args.batch_size} "
        f"params={params:,} load={load_seconds:.1f}s\n"
        f"candidate token len: min={min(lengths)} "
        f"median={statistics.median(lengths)} max={max(lengths)}\n"
        f"rerank per query: median={median:.2f}s "
        f"(warm runs: {[round(t, 2) for t in warm]})\n"
        f"{total_questions} questions -> {total / 60:.1f} min "
        f"({total / 3600:.2f} h) of rerank alone"
    )


if __name__ == "__main__":
    main()
