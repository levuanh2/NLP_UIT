"""Score retrieval on its own: does the context even contain the law the answer cites?

Generation costs ~27s a question; retrieval costs a fraction of that. When the
gap to close is content accuracy rather than answer length, this measures the
ceiling directly — an article the retriever never returned cannot be quoted, no
matter what the generator does.

Coverage is the share of the document numbers ("17/2022/TT-BVHTTDL") and article
numbers ("Điều 25") in the expert answer that appear anywhere in the retrieved
context.

Usage:
  python scripts/coverage_probe.py                       # current .env settings
  DENSE_TOP_K=60 RERANKER_TOP_K=16 python scripts/coverage_probe.py --label wide
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from app.core.config import get_settings
from app.domain.queries import LegalQuery
from app.services.runtime_factory import build_local_rag_runtime

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = re.compile(r"(\d+/\d{4}/[A-ZĐ][A-ZĐ-]*|\d+-\d{4}-[A-ZĐ][A-ZĐ-]*)")
ARTICLE = re.compile(r"Điều\s+(\d+)")


def coverage(reference: str, context: str, pattern: re.Pattern[str]) -> float | None:
    """Share of the reference's citations in the context, None if it cites none."""
    wanted = set(pattern.findall(reference))
    if not wanted:
        return None
    return len(wanted & set(pattern.findall(context))) / len(wanted)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/questions/dev200.json"
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/train/train.json")
    parser.add_argument("--label", default="current")
    parser.add_argument("--limit", type=int, help="Probe only the first N questions.")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/outputs/coverage.jsonl"
    )
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    train = json.loads(args.train.read_text(encoding="utf-8"))
    ids = [q for q in questions if q in train]
    if args.limit:
        ids = ids[: args.limit]

    settings = get_settings()
    runtime = build_local_rag_runtime(settings)
    # The LLM loads with the runtime and is never called here. Idle weights on a
    # free GPU are cheaper than a second copy of the retrieval wiring.
    pipeline = runtime.service.retrieval_pipeline

    documents: list[float] = []
    articles: list[float] = []
    started = time.perf_counter()
    try:
        for position, question_id in enumerate(ids, 1):
            result = pipeline.retrieve(
                LegalQuery(
                    question_id=question_id,
                    question=train[question_id]["question"],
                )
            )
            context = "\n".join(evidence.text for evidence in result.evidences)
            reference = train[question_id]["answer"]
            for pattern, bucket in ((DOCUMENT, documents), (ARTICLE, articles)):
                score = coverage(reference, context, pattern)
                if score is not None:
                    bucket.append(score)
            if position % 25 == 0:
                print(f"  {position}/{len(ids)}", flush=True)
    finally:
        runtime.close()

    row = {
        "label": args.label,
        "questions": len(ids),
        "document_coverage": (
            round(statistics.mean(documents), 4) if documents else None
        ),
        "article_coverage": round(statistics.mean(articles), 4) if articles else None,
        "dense_top_k": settings.dense_top_n,
        "bm25_top_k": settings.bm25_top_n,
        "rrf_top_k": settings.fusion_top_n,
        "reranker_top_k": settings.rerank_top_k,
        "context_max_tokens": settings.context_max_tokens,
        "metadata_min_confidence": settings.metadata_filter_min_confidence,
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
