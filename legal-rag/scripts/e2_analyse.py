"""E2 analysis - RRF/reranker rank blend, retrieval only.

Three runs are compared: the pre-existing baseline artifact, the control run of
the patched code with the blend switched off, and the blend run. The control
exists to prove the patch is inert when the flag is unset; if it does not
reproduce the baseline, nothing else in this file can be trusted.
"""
import json, statistics, sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.retrieval_benchmark import Question, hits

EV = Path("data/evaluation")
PRIOR = EV / "retrieval_enrichment_ab/B-enriched-index-enriched-reranker-k20/per_question.jsonl"
OFF = EV / "e2-off/per_question.jsonl"
ON = EV / "e2-on/per_question.jsonl"
BRK = EV / "step9_failure_breakdown.jsonl"
SQLITE = Path("storage/indexes/v1/metadata/legal.sqlite")

def load(p):
    return {str(json.loads(l)["question_id"]): json.loads(l) for l in p.open(encoding="utf-8")}

prior, off, on = load(PRIOR), load(OFF), load(ON)
brk = load(BRK)
assert set(off) == set(on) == set(prior), "runs cover different question sets"

# ---- control: patched code with the flag unset must equal the prior baseline
drift = [q for q in off
         if off[q]["rrf_reranker"]["child_ids"] != prior[q]["rrf_reranker"]["child_ids"]]
fusion_drift = [q for q in off if off[q]["rrf"]["child_ids"] != prior[q]["rrf"]["child_ids"]]
print(f"CONTROL: blend-off vs prior baseline — reranker differs on {len(drift)}/200, "
      f"fusion differs on {len(fusion_drift)}/200")
if drift or fusion_drift:
    print("  NOTE: control does not reproduce the prior artifact exactly; "
          "comparisons below use e2-off as the baseline, which is the correct "
          "control because it shares the patched code path.")

# ---- resolve child text once, read-only, to recover gold ranks
import sqlite3
need = {cid for d in (off, on) for t in d.values()
        for s in ("dense", "bm25", "rrf", "rrf_reranker") for cid in t[s]["child_ids"]}
con = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
cache, ids = {}, sorted(need)
for i in range(0, len(ids), 900):
    ch = ids[i:i + 900]
    for cid, dn, ar, tx in con.execute(
        f"SELECT child_id, document_name, article, text FROM child_chunks "
        f"WHERE child_id IN ({','.join('?' * len(ch))})", ch):
        cache[cid] = {"document_name": dn or "", "article": ar or "", "text": tx}
con.close()

def gold_rank(cids, q, cutoff=None):
    for i, cid in enumerate(cids if cutoff is None else cids[:cutoff], 1):
        item = cache.get(cid)
        if item and any(hits(item, q)):
            return i
    return None

def stats(run, tag):
    rec, rrf_rank, rr_rank = [], [], []
    lost_rrf = lost_rerank = 0
    per = {}
    for qid, t in run.items():
        b = brk[qid]
        q = Question(qid, "", set(b["gold_documents"]), set(b["gold_articles"]))
        if not q.gold_articles and not q.gold_documents:
            per[qid] = None
            continue
        r_rrf = gold_rank(t["rrf"]["child_ids"], q)
        r_rr = gold_rank(t["rrf_reranker"]["child_ids"], q, 20)
        rec.append(t["rrf_reranker"].get("recall_article@20") or 0.0)
        if r_rrf: rrf_rank.append(r_rrf)
        else: lost_rrf += 1
        if r_rr: rr_rank.append(r_rr)
        else: lost_rerank += 1
        per[qid] = {"rrf_rank": r_rrf, "rerank_rank": r_rr,
                    "recall20": t["rrf_reranker"].get("recall_article@20")}
    print(f"\n=== {tag} ===")
    print(f"  recall_article@20 after rerank   {statistics.mean(rec):.4f}  (n={len(rec)})")
    print(f"  gold lost after RRF              {lost_rrf}")
    print(f"  gold lost after rerank/blend     {lost_rerank}")
    print(f"  median gold rank after RRF       {statistics.median(rrf_rank):.0f}  (n={len(rrf_rank)})")
    print(f"  median gold rank after rerank    {statistics.median(rr_rank):.0f}  (n={len(rr_rank)})")
    return per, statistics.mean(rec)

per_off, rec_off = stats(off, "BASELINE (e2-off, blend disabled)")
per_on, rec_on = stats(on, "E2 (blend RRF k=60)")
print(f"\nDELTA recall_article@20: {rec_on - rec_off:+.4f}")

# ---- per-question movement
better = worse = same = 0
for qid in off:
    a, b = per_off.get(qid), per_on.get(qid)
    if not a or not b: continue
    ra, rb = a["recall20"] or 0.0, b["recall20"] or 0.0
    if rb > ra + 1e-9: better += 1
    elif rb < ra - 1e-9: worse += 1
    else: same += 1
print(f"per-question recall@20:  improved {better}   regressed {worse}   unchanged {same}")

# ---- the 7 known reranker_failure questions
rf = [q for q in brk if brk[q]["cls"] == "reranker_failure"]
print(f"\n=== the {len(rf)} known reranker_failure questions ===")
print(f"  {'qid':<9}{'bm25':>6}{'rrfRank':>9}{'baseRerank':>12}{'E2Rerank':>10}"
      f"{'baseRec':>9}{'E2Rec':>7}  rescued")
rescued = 0
for qid in sorted(rf):
    b = brk[qid]
    q = Question(qid, "", set(b["gold_documents"]), set(b["gold_articles"]))
    bm = gold_rank(off[qid]["bm25"]["child_ids"], q) if "bm25" in off[qid] else None
    a, c = per_off[qid], per_on[qid]
    ok = (c["recall20"] or 0) > (a["recall20"] or 0)
    rescued += ok
    print(f"  {qid:<9}{str(bm):>6}{str(a['rrf_rank']):>9}{str(a['rerank_rank']):>12}"
          f"{str(c['rerank_rank']):>10}{a['recall20'] or 0:>9.2f}{c['recall20'] or 0:>7.2f}"
          f"  {'YES' if ok else 'no'}")
print(f"  rescued {rescued}/{len(rf)}")

rows = []
for qid in off:
    a, b = per_off.get(qid), per_on.get(qid)
    rows.append({"question_id": qid, "cls": brk[qid]["cls"],
                 "gold_articles": brk[qid]["gold_articles"],
                 "gold_documents": brk[qid]["gold_documents"],
                 "baseline": a, "e2": b})
Path("data/evaluation/e2_reranker_blend_per_question.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
print("\nwrote data/evaluation/e2_reranker_blend_per_question.jsonl")
