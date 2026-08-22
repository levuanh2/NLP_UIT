import json
from pathlib import Path

path = Path("data/evaluation/retrieval_enrichment_ab/B-enriched-index-enriched-reranker-k20/per_question.jsonl")
if path.is_file():
    with path.open("r", encoding="utf-8") as f:
        q = json.loads(f.readline())
        print("Lengths:")
        for s in ('dense', 'bm25', 'rrf', 'rrf_reranker'):
            print(f"  {s}: {len(q[s]['child_ids'])}")
else:
    print("Trace file not found")
