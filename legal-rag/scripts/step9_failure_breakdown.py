"""PHASE 2 - detailed failure breakdown for the frozen v5 config.

Retrieval is NOT re-run. The candidate ids already stored in the enriched/k20
trace are resolved against the metadata SQLite opened READ-ONLY (mode=ro), so
nothing can write to the production index. The benchmark's own hit logic is
imported rather than re-implemented, and the recomputed recall@20 is checked
against the recall the trace already stores; a mismatch aborts instead of
reporting invented ranks.
"""
import json, sqlite3, statistics, sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.retrieval_benchmark import Question, hits

AB = Path("data/evaluation/retrieval_enrichment_ab")
TRACE = AB / "B-enriched-index-enriched-reranker-k20/per_question.jsonl"
CLS = Path("data/evaluation/step8_per_question.jsonl")
WINNER = Path("data/outputs/dev200-enriched-k20-ckpt350/submission.json")
SQLITE = Path("storage/indexes/v1/metadata/legal.sqlite")
STAGES = ["dense", "bm25", "rrf", "rrf_reranker"]

trace = [json.loads(l) for l in TRACE.open(encoding="utf-8")]
trace = {str(t["question_id"]): t for t in trace}
cls = {json.loads(l)["question_id"]: json.loads(l) for l in CLS.open(encoding="utf-8")}
sub = json.loads(WINNER.read_text(encoding="utf-8"))
train = json.loads(Path("data/train/train.json").read_text(encoding="utf-8"))

need = {cid for t in trace.values() for s in STAGES for cid in t[s]["child_ids"]}
con = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
cache = {}
ids = sorted(need)
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    qmarks = ",".join("?" * len(chunk))
    for cid, dname, art, txt in con.execute(
        f"SELECT child_id, document_name, article, text FROM child_chunks "
        f"WHERE child_id IN ({qmarks})", chunk):
        cache[cid] = {"document_name": dname or "", "article": art or "", "text": txt}
con.close()
print(f"needed {len(need)} child ids, resolved {len(cache)}, missing {len(need)-len(cache)}")

rows, mismatches = [], 0
for qid, t in trace.items():
    rec = train.get(qid, {})
    q = Question(qid, rec.get("question", ""),
                 set(t["gold_documents"]), set(t["gold_articles"]))
    per_stage = {}
    for s in STAGES:
        cids = t[s]["child_ids"]
        rank, full_a, cap_a = None, set(), set()
        for i, cid in enumerate(cids, 1):
            item = cache.get(cid)
            if item is None:
                continue
            d, a = hits(item, q)
            if (d or a) and rank is None:
                rank = i
            full_a |= a
            if i <= 20:
                cap_a |= a
        cap = (len(cap_a) / len(q.gold_articles)) if q.gold_articles else None
        stored = t[s].get("recall_article@20")
        if stored is not None and cap is not None and abs(cap - stored) > 1e-6:
            mismatches += 1
        per_stage[s] = {"first_gold_rank": rank, "returned": len(cids),
                        "recall_article_full": (len(full_a) / len(q.gold_articles))
                                                if q.gold_articles else None,
                        "recall_article@20_recomputed": cap,
                        "recall_article@20_stored": stored,
                        "total_ms": t[s]["total_ms"]}

    c = cls.get(qid, {})
    rows.append({"question_id": qid, "cls": c.get("cls"), "failure_stage": t["failure_stage"],
                 "question": q.question, "gold_documents": sorted(q.gold_documents),
                 "gold_articles": sorted(q.gold_articles),
                 "meteor": c.get("meteor"), "rouge_l": c.get("rouge_l"),
                 "pred_words": c.get("pred_words"), "ref_words": c.get("ref_words"),
                 "stages": per_stage, "metadata_filter": t["metadata_filter"],
                 "pred_head": sub.get(qid, {}).get("answer", "")[:200],
                 "ref_head": (rec.get("answer") or "")[:200]})

print(f"recall mismatches vs stored trace: {mismatches}")
if mismatches:
    print("ABORT: recomputed recall disagrees with the trace; ranks not trustworthy")
    raise SystemExit(2)

Path("data/evaluation/step9_failure_breakdown.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

def summarise(name):
    xs = [r for r in rows if r["cls"] == name]
    if not xs: return
    ms = [r["meteor"] for r in xs if r["meteor"] is not None]
    print(f"\n=== {name}  n={len(xs)}  mean METEOR {statistics.mean(ms):.4f} ===")
    for s in STAGES:
        got = [r["stages"][s]["first_gold_rank"] for r in xs
               if r["stages"][s]["first_gold_rank"]]
        line = f"  {s:<14} gold found {len(got):>3}/{len(xs)}"
        if got:
            line += (f"  median rank {statistics.median(got):.0f}"
                     f"  min {min(got)}  max {max(got)}")
        print(line)
    ng = sum(1 for r in xs if not r["gold_articles"] and not r["gold_documents"])
    print(f"  no parseable gold citation: {ng}/{len(xs)}")
    print(f"  median gold articles {statistics.median([len(r['gold_articles']) for r in xs]):.0f}"
          f"  median gold docs {statistics.median([len(r['gold_documents']) for r in xs]):.0f}")
    print(f"  median words ref {statistics.median([r['ref_words'] for r in xs]):.0f}"
          f"  pred {statistics.median([r['pred_words'] for r in xs]):.0f}")

for n in ["both_miss", "reranker_failure", "fusion_failure", "generation_failure",
          "verbosity_failure", "ok"]:
    summarise(n)
