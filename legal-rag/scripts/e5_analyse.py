"""E5 - retrieval score diagnostics. ANALYSIS ONLY, no LLM, no generation.

Every number here is read from scores the pipeline actually computed and the
instrumented benchmark persisted. Nothing is reconstructed from rank, and a
field the artifact does not carry is reported as "not available" rather than
imputed.

Vocabulary, fixed once so the tables below cannot drift:
  pool          the 50 fused candidates the cross-encoder scores
  rerank_rank   position in that pool ordered by cross-encoder score, 1..50
  cut           RERANKER_TOP_K = 20; gold at rerank_rank > 20 is dropped
  lost          gold exists, is not in the returned top-20 after reranking
"""
import json
import statistics
from collections import Counter
from pathlib import Path

EV = Path("data/evaluation")
E5 = EV / "e5-full/per_question.jsonl"
BRK = EV / "step9_failure_breakdown.jsonl"
CUT = 20

e5 = {str(json.loads(l)["question_id"]): json.loads(l) for l in E5.open(encoding="utf-8")}
brk = {str(json.loads(l)["question_id"]): json.loads(l) for l in BRK.open(encoding="utf-8")}
assert set(e5) == set(brk), "e5 run and failure breakdown cover different questions"
N = len(e5)


def gold_rank(row, stage):
    """First gold position in a stage's candidate list, from the persisted
    is_gold flag. None if the stage returned no gold."""
    for c in row[stage].get("candidates", []):
        if c.get("is_gold"):
            return c["rank"]
    return None


def cut_score(row):
    """Score of the last candidate that survived the top-20 cut - the bar a
    dropped gold had to clear. Read, not inferred."""
    cands = row["rrf_reranker"].get("candidates", [])
    scored = [c for c in cands if c.get("rerank_score") is not None]
    return scored[-1]["rerank_score"] if scored else None


rows = []
for qid, r in e5.items():
    b = brk[qid]
    rr = r["rrf_reranker"]
    has_gold = bool(b["gold_articles"] or b["gold_documents"])
    grank = rr.get("gold_best_rerank_rank")
    in_pool = bool(rr.get("gold_in_rerank_pool"))
    after = gold_rank(r, "rrf_reranker")

    if not has_gold:
        cls = "E no parseable gold"
    elif not in_pool:
        cls = "A gold never entered the pool"
    elif after is not None:
        cls = "kept"
    elif grank <= 25:
        cls = "B near-miss at the cut"
    elif grank <= 40:
        cls = "C mid-pool demotion"
    else:
        cls = "D hard rejection"

    rows.append({
        "question_id": qid,
        "cls": b["cls"],
        "e5_class": cls,
        "meteor": b["meteor"],
        "gold_articles": b["gold_articles"],
        "gold_documents": b["gold_documents"],
        "has_parseable_gold": has_gold,
        "gold_rank_dense": gold_rank(r, "dense"),
        "gold_rank_bm25": gold_rank(r, "bm25"),
        "gold_rank_rrf": gold_rank(r, "rrf"),
        "gold_rank_after_rerank": after,
        "gold_in_rerank_pool": in_pool,
        "gold_best_rerank_rank": grank,
        "gold_best_rerank_score": rr.get("gold_best_rerank_score"),
        "gold_best_child_id": rr.get("gold_best_child_id"),
        "rerank_top1_score": rr.get("rerank_top1_score"),
        "rerank_top2_score": rr.get("rerank_top2_score"),
        "rerank_margin_top1_top2": rr.get("rerank_margin_top1_top2"),
        "gold_score_gap_to_top1": rr.get("gold_score_gap_to_top1"),
        "rerank_cut_score_at_20": cut_score(r),
        "blockers_above_gold": rr.get("blockers_above_gold", []),
        "rerank_scored_count": rr.get("rerank_scored_count"),
    })
by_id = {r["question_id"]: r for r in rows}

print(f"=== E5 diagnostics: {N} questions, pool size "
      f"{sorted({r['rerank_scored_count'] for r in rows})} ===")

lost = [r for r in rows if r["has_parseable_gold"] and r["gold_rank_after_rerank"] is None]
print(f"gold present but lost after rerank: {len(lost)}")
print("e5 class breakdown of those:", dict(Counter(r["e5_class"] for r in lost)))


def table(title, ids):
    print(f"\n=== {title} ({len(ids)}) ===")
    print(f"  {'qid':<8}{'bm25':>6}{'rrf':>5}{'pool':>6}{'goldSc':>9}{'top1':>8}"
          f"{'top2':>8}{'margin':>8}{'cut@20':>9}{'gap':>8}  class")
    for qid in ids:
        r = by_id[qid]
        f = lambda v, w, p=2: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'n/a':>{w}}")
        print(f"  {qid:<8}{str(r['gold_rank_bm25']):>6}{str(r['gold_rank_rrf']):>5}"
              f"{str(r['gold_best_rerank_rank']):>6}{f(r['gold_best_rerank_score'],9)}"
              f"{f(r['rerank_top1_score'],8)}{f(r['rerank_top2_score'],8)}"
              f"{f(r['rerank_margin_top1_top2'],8)}{f(r['rerank_cut_score_at_20'],9)}"
              f"{f(r['gold_score_gap_to_top1'],8)}  {r['e5_class']}")


for target in ("reranker_failure", "fusion_failure", "both_miss"):
    ids = sorted(q for q in by_id if by_id[q]["cls"] == target)
    table(target, ids)

# ---- what blocked gold on the 7, by identity not just by score
print("\n=== reranker_failure: what outranked gold (nearest 3 blockers) ===")
for qid in sorted(q for q in by_id if by_id[q]["cls"] == "reranker_failure"):
    r = by_id[qid]
    print(f"  {qid}  gold {r['gold_best_child_id']} "
          f"score {r['gold_best_rerank_score']} at pool rank {r['gold_best_rerank_rank']}")
    for bl in r["blockers_above_gold"]:
        print(f"      rank {bl['rank']:>3}  score {bl['score']:>10.4f}  {bl['child_id']}")

# ---- PHASE 5 Q1: is the reranker the bottleneck?
print("\n=== Q1 where is gold actually lost ===")
have = [r for r in rows if r["has_parseable_gold"]]
not_in_pool = [r for r in have if not r["gold_in_rerank_pool"]]
in_pool_lost = [r for r in have if r["gold_in_rerank_pool"] and r["gold_rank_after_rerank"] is None]
kept = [r for r in have if r["gold_rank_after_rerank"] is not None]
print(f"  questions with parseable gold      {len(have)}")
print(f"  gold never reached the pool        {len(not_in_pool)}   (upstream: dense+bm25+RRF)")
print(f"  gold in pool, dropped by reranker  {len(in_pool_lost)}   (reranker's own doing)")
print(f"  gold kept in top-20                {len(kept)}")

# ---- PHASE 5 Q2/Q3: is there a score/margin pattern that separates them?
def stat(name, vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f"  {name:<34} not available")
        return
    q = statistics.quantiles(vals, n=4) if len(vals) > 3 else [float("nan")] * 3
    print(f"  {name:<34} n={len(vals):<4} median {statistics.median(vals):>8.3f}"
          f"  q1 {q[0]:>8.3f}  q3 {q[2]:>8.3f}  min {min(vals):>8.3f}  max {max(vals):>8.3f}")

print("\n=== Q2/Q3 score distributions ===")
print(" gold score, kept vs dropped:")
stat("kept: gold rerank score", [r["gold_best_rerank_score"] for r in kept])
stat("dropped: gold rerank score", [r["gold_best_rerank_score"] for r in in_pool_lost])
stat("kept: gap to top1", [r["gold_score_gap_to_top1"] for r in kept])
stat("dropped: gap to top1", [r["gold_score_gap_to_top1"] for r in in_pool_lost])
print(" the bar gold had to clear:")
stat("cut score at rank 20 (all)", [r["rerank_cut_score_at_20"] for r in have])
stat("top1 score (all)", [r["rerank_top1_score"] for r in have])
stat("top1-top2 margin (all)", [r["rerank_margin_top1_top2"] for r in have])
stat("top1-top2 margin, gold kept", [r["rerank_margin_top1_top2"] for r in kept])
stat("top1-top2 margin, gold dropped", [r["rerank_margin_top1_top2"] for r in in_pool_lost])

print("\n pool rank of gold among the dropped:")
ranks = sorted(r["gold_best_rerank_rank"] for r in in_pool_lost)
print(f"  {ranks}")
for k in (20, 25, 30, 40, 50):
    n = sum(1 for r in in_pool_lost if r["gold_best_rerank_rank"] <= k)
    print(f"  raising RERANKER_TOP_K to {k:>3} would recover {n:>2}/{len(in_pool_lost)} of them")

# ---- PHASE 5 Q4: could a score threshold separate dropped gold from noise?
print("\n=== Q4 can a score rule separate gold from what outranks it ===")
print("  A threshold only works if dropped gold scores ABOVE the junk that")
print("  displaced it. Compare dropped-gold score against the cut score:")
above = [r for r in in_pool_lost
         if r["gold_best_rerank_score"] is not None
         and r["rerank_cut_score_at_20"] is not None
         and r["gold_best_rerank_score"] >= r["rerank_cut_score_at_20"]]
print(f"  dropped gold scoring >= the rank-20 cut score: {len(above)}/{len(in_pool_lost)}")
print("  (a gold below the cut score is, by construction, one the cross-encoder")
print("   ranked worse than 20 other candidates - no monotone threshold on that")
print("   same score can rescue it without also admitting all 20)")

# ---- PHASE 5 Q5/Q6: upper bound on dev200 METEOR
print("\n=== Q5/Q6 upper bound on dev200 METEOR ===")
healthy = [brk[q]["meteor"] for q in brk if brk[q]["cls"] == "ok"]
h = statistics.mean(healthy)
print(f"  ASSUMPTION (stated, not measured): a question whose gold is retrieved")
print(f"  scores the mean of the healthy 'ok' class, {h:.4f}. This is generous -")
print(f"  these are the hardest questions in the set - so it is an upper bound.")
for label, group in (("reranker-dropped gold only (in pool, recoverable)", in_pool_lost),
                     ("all reranker_failure + fusion_failure", [r for r in rows if r["cls"] in ("reranker_failure", "fusion_failure")]),
                     ("every question with gold lost anywhere", [r for r in have if r["gold_rank_after_rerank"] is None])):
    gain = sum(max(0.0, h - r["meteor"]) for r in group) / N
    print(f"  {label:<52} n={len(group):>3}  +{gain:.4f}  ({gain/0.028:.2f}x SE)")

out = EV / "e5_retrieval_score_diagnostics.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
               encoding="utf-8")
print(f"\nwrote {out}")
