"""Ask whether a larger reranker top-k survives the context budget.

The retrieval benchmark stops at the reranker's output. Two stages still stand
between that output and the prompt: the parent expander swaps each child chunk
for the parent that contains it, and the context builder drops whatever does not
fit CONTEXT_MAX_TOKENS. Evidence rescued at rank 11-20 is only worth having if
it is still there when the prompt is written.

Read-only. Nothing is configured, published, or written outside --output.

Usage:
  MODEL_DEVICE=cpu python scripts/context_budget_probe.py --limit 60
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_benchmark import (  # noqa: E402
    BenchmarkError,
    hits,
    load_questions,
    read_index_state,
)

from app.core.config import get_settings  # noqa: E402
from app.services.runtime_factory import build_local_rag_runtime  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOP_KS = (10, 20, 40)


def gold_survives(evidences: list[Any], question: Any) -> bool:
    """Is any gold citation still present after expansion and budgeting?"""
    for evidence in evidences:
        item = {
            "document_name": evidence.document_name or "",
            "article": evidence.article or "",
            "text": evidence.text,
        }
        if any(hits(item, question)):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/questions/dev200.json"
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/train/train.json")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/evaluation/context_budget.json"
    )
    args = parser.parse_args()

    settings = get_settings()
    index_root = Path(settings.index_root_dir)
    if not index_root.is_absolute():
        index_root = (ROOT / index_root).resolve()
    index = read_index_state(index_root)
    questions = load_questions(args.questions, args.train, args.limit)

    print(f"INDEX: {index['current']} ({index['manifest']['child_count']} children)")
    print(f"QUESTIONS: {len(questions)}")
    print(
        f"BUDGET: context_max_tokens={settings.context_max_tokens} "
        f"neighbor_window={settings.context_neighbor_window} "
        f"max_parents_per_document={settings.max_parents_per_document}"
    )

    runtime = build_local_rag_runtime(settings)
    pipeline = runtime.service.retrieval_pipeline
    samples: dict[int, list[dict[str, Any]]] = {top_k: [] for top_k in TOP_KS}
    try:
        for position, question in enumerate(questions, 1):
            dense = pipeline.dense_retriever.retrieve(
                question.question, candidate_ids=None, top_k=pipeline.dense_top_k
            )
            bm25 = pipeline.bm25_retriever.retrieve(
                question.question, candidate_ids=None, top_k=pipeline.bm25_top_k
            )
            fused = pipeline.fusion.fuse(
                dense, bm25, k=pipeline.rrf_k, top_k=pipeline.rrf_top_k
            )
            for top_k in TOP_KS:
                ranked = pipeline.reranker.rerank(
                    question.question, fused, top_k=top_k
                )
                expanded = pipeline.parent_expander.expand(ranked)
                context = pipeline.context_builder.build(question.question, expanded)
                samples[top_k].append(
                    {
                        "question_id": question.question_id,
                        "children": len(ranked),
                        "parents_after_expansion": len(expanded),
                        "evidences_in_context": len(context.evidences),
                        "dropped_by_budget": len(expanded) - len(context.evidences),
                        "token_count": context.token_count or 0,
                        "gold_after_rerank": any(
                            gold_survives([e], question) for e in expanded
                        ),
                        "gold_in_context": gold_survives(context.evidences, question),
                    }
                )
            if position % 20 == 0:
                print(f"  {position}/{len(questions)}", flush=True)
    finally:
        runtime.close()

    report: dict[str, Any] = {
        "context_max_tokens": settings.context_max_tokens,
        "max_parents_per_document": settings.max_parents_per_document,
        "neighbor_window": settings.context_neighbor_window,
        "questions": len(questions),
        "by_top_k": {},
    }
    for top_k, rows in samples.items():
        tokens = [row["token_count"] for row in rows]
        parents = [row["parents_after_expansion"] for row in rows]
        kept = [row["evidences_in_context"] for row in rows]
        truncated = [row for row in rows if row["dropped_by_budget"] > 0]
        lost_gold = [
            row
            for row in rows
            if row["gold_after_rerank"] and not row["gold_in_context"]
        ]
        report["by_top_k"][str(top_k)] = {
            "parents_after_expansion": {
                "mean": round(statistics.mean(parents), 2),
                "median": statistics.median(parents),
                "max": max(parents),
            },
            "evidences_in_context": {
                "mean": round(statistics.mean(kept), 2),
                "median": statistics.median(kept),
                "max": max(kept),
            },
            "token_count": {
                "mean": round(statistics.mean(tokens), 1),
                "median": statistics.median(tokens),
                "max": max(tokens),
                "at_or_over_budget": sum(
                    1 for t in tokens if t >= settings.context_max_tokens
                ),
            },
            "questions_truncated_by_budget": len(truncated),
            "questions_with_gold_in_context": sum(
                1 for row in rows if row["gold_in_context"]
            ),
            "gold_lost_to_budget": [row["question_id"] for row in lost_gold],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**report, "rows": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    for top_k in TOP_KS:
        stats = report["by_top_k"][str(top_k)]
        print(f"top_k={top_k}")
        print(
            f"  parents after expansion: median "
            f"{stats['parents_after_expansion']['median']} "
            f"(max {stats['parents_after_expansion']['max']})"
        )
        print(
            f"  evidences reaching the prompt: median "
            f"{stats['evidences_in_context']['median']} "
            f"(max {stats['evidences_in_context']['max']})"
        )
        print(
            f"  context tokens: median {stats['token_count']['median']} "
            f"max {stats['token_count']['max']}, "
            f"{stats['token_count']['at_or_over_budget']} question(s) at the cap"
        )
        print(f"  truncated by budget: {stats['questions_truncated_by_budget']}")
        print(f"  gold present in prompt: {stats['questions_with_gold_in_context']}")
        print(f"  gold lost to the budget: {len(stats['gold_lost_to_budget'])}")
    print()
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
