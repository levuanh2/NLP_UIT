"""Time Vi-Qwen2-3B-RAG answer generation on a realistic grounded prompt.

Usage: python scripts/bench_generate.py [--device cpu|cuda]
       [--dtype auto|float16|bfloat16] [--quantization none|nf4] [--runs 2]
"""

import argparse
import sqlite3
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "storage/index-builds-qualified/v1/metadata/legal.sqlite"
QUESTION = "Mẫu thông báo thay đổi người đại diện theo pháp luật gồm những nội dung gì?"
TOTAL_QUESTIONS = 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--context-tokens", type=int, default=3500)
    parser.add_argument("--quantization", default="none", choices=["none", "nf4"])
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    chunks = [
        row[0]
        for row in conn.execute(
            "SELECT text FROM child_chunks "
            "WHERE rowid % 997 = 0 AND token_count >= 200 LIMIT 40"
        )
    ]
    conn.close()

    from app.generation.llm.factory import LLMGeneratorFactory

    generator = LLMGeneratorFactory.create(
        provider="local_transformers",
        model_name="AITeamVN/Vi-Qwen2-3B-RAG",
        device=args.device,
        dtype=args.dtype,
        local_files_only=True,
        trust_remote_code=False,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        min_new_tokens=1,
        repetition_penalty=1.0,
        quantization=args.quantization,
    )
    load_started = time.perf_counter()
    generator.load()
    load_seconds = time.perf_counter() - load_started

    # Grow the evidence block until it reaches the target prompt size.
    context = ""
    for chunk in chunks:
        if generator.count_tokens(context) >= args.context_tokens:
            break
        context += chunk + "\n\n"
    prompt = (
        "SYSTEM:\nBạn là trợ lý pháp lý. Chỉ trả lời dựa trên CONTEXT.\n"
        f"CONTEXT:\n{context}\n### Câu hỏi\n{QUESTION}\n"
    )

    per_run: list[float] = []
    for index in range(args.runs):
        started = time.perf_counter()
        generator.generate(prompt)
        elapsed = time.perf_counter() - started
        metrics = generator.last_generation_metrics
        per_run.append(elapsed)
        print(
            f"  run {index + 1}: {elapsed:.1f}s "
            f"(in={metrics.input_tokens} out={metrics.generated_tokens} "
            f"{metrics.tokens_per_second:.1f} tok/s)"
        )

    warm = per_run[1:] or per_run
    median = statistics.median(warm)
    total = median * TOTAL_QUESTIONS
    print(
        f"\ndevice={generator._resolved_device} dtype={args.dtype} "
        f"quantization={args.quantization} load={load_seconds:.1f}s\n"
        f"generate per question: median={median:.1f}s\n"
        f"{TOTAL_QUESTIONS} questions -> {total / 3600:.1f} h of generation alone"
    )
    generator.unload()


if __name__ == "__main__":
    main()
