"""STEP 8 - retrieval failure analysis for the frozen v5 config.

Reads only existing artifacts: the enriched/k20 retrieval trace, the winner's
dev200 answers, the context-budget probe, and train.json for references.
Nothing is re-run, no index is touched, no gold label is invented.
"""
import json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from app.evaluation.generation_metrics import meteor, rouge_l

ROOT = Path(".")
AB = ROOT / "data/evaluation/retrieval_enrichment_ab"
TRACE = AB / "B-enriched-index-enriched-reranker-k20/per_question.jsonl"
BUDGET = AB / "context_budget_enriched.json"
WINNER = ROOT / "data/outputs/dev200-enriched-k20-ckpt350/submission.json"
TRAIN = ROOT / "data/train/train.json"
K = "20"

# Thresholds are analysis choices, not measurements. They are stated in the
# report so a reader can re-cut the same data differently.
POOR_METEOR = 0.30      # below this the answer is treated as not carrying the answer
VERBOSE_RATIO = 2.0     # predicted/reference word ratio that counts as padded

trace = [json.loads(l) for l in TRACE.open(encoding="utf-8")]
budget = json.loads(BUDGET.read_text(encoding="utf-8"))
sub = json.loads(WINNER.read_text(encoding="utf-8"))
train = json.loads(TRAIN.read_text(encoding="utf-8"))

bk = budget["by_top_k"][K]
gold_lost = set(str(q) for q in bk.get("gold_lost_to_budget", []))
truncated = bk.get("questions_truncated_by_budget", 0)

rows = []
for t in trace:
    qid = str(t["question_id"])
    rec = sub.get(qid)
    ref = train.get(qid, {}).get("answer")
    pred = rec["answer"] if rec else None
    m = r = None
    plen = rlen = 0
    if pred is not None and ref is not None:
        m, r = meteor(pred, ref), rouge_l(pred, ref)
        plen, rlen = len(pred.split()), len(ref.split())
    ratio = (plen / rlen) if rlen else None

    mf = t["metadata_filter"]
    has_identifier = bool(mf.get("document_number") or mf.get("document_name"))
    stage = t["failure_stage"]

    # One bucket per question, most upstream cause first: a question that never
    # retrieved its gold cannot also be a generation failure.
    if stage == "both_miss":
        cls = "metadata_entity_failure" if has_identifier else "both_miss"
    elif stage == "rrf_ranking":
        cls = "fusion_failure"
    elif stage == "reranker_ranking":
        cls = "reranker_failure"
    elif qid in gold_lost:
        cls = "context_failure"
    elif m is None:
        cls = "unscored"
    elif m < POOR_METEOR:
        cls = "generation_failure"
    elif ratio is not None and ratio > VERBOSE_RATIO:
        cls = "verbosity_failure"
    else:
        cls = "ok"

    rows.append(dict(question_id=qid, cls=cls, failure_stage=stage,
                     has_identifier=has_identifier, meteor=m, rouge_l=r,
                     pred_words=plen, ref_words=rlen, ratio=ratio,
                     dense_hit20=t["dense"].get("recall_article@20", 0),
                     bm25_hit20=t["bm25"].get("recall_article@20", 0),
                     rrf_hit20=t["rrf"].get("recall_article@20", 0),
                     rerank_hit20=t["rrf_reranker"].get("recall_article@20", 0)))

counts = Counter(r["cls"] for r in rows)
order = ["both_miss", "fusion_failure", "reranker_failure", "context_failure",
         "generation_failure", "verbosity_failure", "metadata_entity_failure",
         "ok", "unscored"]

def agg(cls):
    xs = [r for r in rows if r["cls"] == cls]
    ms = [r["meteor"] for r in xs if r["meteor"] is not None]
    return (len(xs), statistics.mean(ms) if ms else None,
            statistics.median([r["pred_words"] for r in xs]) if xs else None,
            [r["question_id"] for r in xs][:5])

out = {"source_trace": str(TRACE), "winner_run": str(WINNER), "n": len(rows),
       "top_k": int(K), "thresholds": {"poor_meteor": POOR_METEOR,
                                       "verbose_ratio": VERBOSE_RATIO},
       "context_budget": {"questions_truncated_by_budget": truncated,
                          "gold_lost_to_budget": sorted(gold_lost)},
       "counts": {c: counts.get(c, 0) for c in order},
       "classes": {}}
for c in order:
    n, mm, mw, ids = agg(c)
    out["classes"][c] = {"count": n, "pct": round(100 * n / len(rows), 1),
                         "mean_meteor": round(mm, 4) if mm is not None else None,
                         "median_pred_words": mw, "example_ids": ids}
overall = [r["meteor"] for r in rows if r["meteor"] is not None]
out["overall_mean_meteor"] = round(statistics.mean(overall), 4)
out["scored"] = len(overall)

Path("data/evaluation/step8_retrieval_failure_analysis.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
with Path("data/evaluation/step8_per_question.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"n={len(rows)} scored={len(overall)} overall METEOR={out['overall_mean_meteor']}")
print(f"{'class':<26}{'n':>5}{'pct':>7}{'meanMETEOR':>12}{'medWords':>10}")
for c in order:
    d = out["classes"][c]
    if d["count"] == 0: continue
    print(f"{c:<26}{d['count']:>5}{d['pct']:>7}{str(d['mean_meteor']):>12}{str(d['median_pred_words']):>10}")
print("context: truncated_by_budget=%s gold_lost=%s" % (truncated, sorted(gold_lost)))
