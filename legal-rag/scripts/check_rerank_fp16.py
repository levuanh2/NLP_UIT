"""Check fp16-on-CUDA reranker keeps the same top-k as fp32-on-CPU."""

import sqlite3
import time

from app.domain.retrieval import RetrievalCandidate
from app.retrieval.reranking.factory import RerankerFactory

QUERY = "Mẫu thông báo thay đổi người đại diện theo pháp luật"

conn = sqlite3.connect("storage/index-builds-qualified/v1/metadata/legal.sqlite")
rows = list(
    conn.execute(
        "SELECT child_id, text FROM child_chunks "
        "WHERE rowid % 997 = 0 AND token_count >= 200 LIMIT 30"
    )
)
conn.close()
candidates = [
    RetrievalCandidate(child_id=cid, text=text, score=0.0, source="rrf", rank=i + 1)
    for i, (cid, text) in enumerate(rows)
]

top: dict[str, list[str]] = {}
for device in ("cuda", "cpu"):
    reranker = RerankerFactory.create(
        "local_transformers",
        "AITeamVN/Vietnamese_Reranker",
        device,
        True,
        False,
        parameter_budget_approved=True,
    )
    reranker.load()
    dtype = next(reranker._model.parameters()).dtype
    reranker.rerank(QUERY, candidates, top_k=5)  # warmup
    started = time.perf_counter()
    result = reranker.rerank(QUERY, candidates, top_k=5)
    elapsed = time.perf_counter() - started
    top[device] = [candidate.child_id for candidate in result]
    print(f"{device} dtype={dtype} {elapsed:.2f}s top5={top[device]}")
    reranker.unload()

assert top["cuda"] == top["cpu"], f"fp16 changed ranking: {top}"
print("OK: fp16 cuda top-5 matches fp32 cpu")
