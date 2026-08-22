"""E4 - adaptive context analysis. ANALYSIS ONLY, no inference, no GPU.

Every feature here is computable from retrieval/reranking output BEFORE the LLM
is called. Gold-derived fields in the trace (first_hit_rank, any_hit,
document_coverage, recall_*, fused_*, rank_change, gold_in_context) are excluded
from the feature set by construction; they are used only to label, never to gate.
"""
import json, statistics
from collections import Counter
from pathlib import Path

EV = Path("data/evaluation")
TRACE = EV / "e2-off/per_question.jsonl"
E1 = EV / "e1_per_question.jsonl"
BUDGET = EV / "retrieval_enrichment_ab/context_budget_enriched.json"

trace = {str(json.loads(l)["question_id"]): json.loads(l) for l in TRACE.open(encoding="utf-8")}
e1 = {json.loads(l)["question_id"]: json.loads(l) for l in E1.open(encoding="utf-8")}
budget = json.loads(BUDGET.read_text(encoding="utf-8"))
brows = {str(r["question_id"]): r for r in budget["rows"]["20"]}
print(f"context probe was built with neighbor_window={budget['neighbor_window']}, "
      f"context_max_tokens={budget['context_max_tokens']}, "
      f"max_parents_per_document={budget['max_parents_per_document']}")

def article_of(cid):
    p = cid.split(":")
    return (p[1], p[4]) if len(p) > 4 else (p[1], None)

K = 20
feat = {}
for qid, t in trace.items():
    top = t["rrf_reranker"]["child_ids"][:K]
    rrf = t["rrf"]["child_ids"][:K]
    dense = set(t["dense"]["child_ids"][:K])
    bm25 = set(t["bm25"]["child_ids"][:K])
    arts = [article_of(c) for c in top]
    counts = Counter(arts)
    rrf_pos = {c: i for i, c in enumerate(t["rrf"]["child_ids"], 1)}
    disp = [abs(rrf_pos[c] - i) for i, c in enumerate(top, 1) if c in rrf_pos]
    b = brows.get(qid, {})
    mf = t["metadata_filter"]
    feat[qid] = {
        "n_articles_top20": len(counts),
        "n_documents_top20": len({a[0] for a in arts}),
        "max_chunks_per_article": max(counts.values()) if counts else 0,
        "top1_article_share": (counts[arts[0]] / len(arts)) if arts else 0.0,
        "dense_bm25_overlap": len(dense & bm25) / K,
        "rrf_kept_in_top20": len(set(top) & set(rrf)) / K,
        "mean_rank_displacement": statistics.mean(disp) if disp else 0.0,
        "children": b.get("children"),
        "parents_after_expansion": b.get("parents_after_expansion"),
        "evidences_in_context": b.get("evidences_in_context"),
        "token_count": b.get("token_count"),
        "dropped_by_budget": b.get("dropped_by_budget"),
        "metadata_confidence": mf.get("confidence"),
        "metadata_candidate_count": mf.get("candidate_count"),
        "bm25_ms": t["bm25"]["total_ms"],
        "reranker_ms": t["rrf_reranker"]["parts"].get("reranker_ms"),
    }

# Label: did narrowing the context actually help this question? That is the
# thing a gate would need to predict, and it is NOT the failure class.
for qid in feat:
    r = e1[qid]
    feat[qid]["_cls"] = r["cls"]
    feat[qid]["_delta"] = r["e1_meteor"] - r["base_meteor"]
    feat[qid]["_e1_helps"] = r["e1_meteor"] > r["base_meteor"]

FEATURES = [k for k in next(iter(feat.values())) if not k.startswith("_")]
avail = [f for f in FEATURES if all(feat[q][f] is not None for q in feat)]
missing = [f for f in FEATURES if f not in avail]
print(f"\nfeatures computable for all 200: {len(avail)}")
if missing:
    print(f"features NOT available for every question: {missing}")

groups = {
    "A generation_failure": lambda f: f["_cls"] == "generation_failure",
    "B healthy(ok+verbose)": lambda f: f["_cls"] in ("ok", "verbosity_failure"),
    "C both_miss": lambda f: f["_cls"] == "both_miss",
    "D rerank+fusion fail": lambda f: f["_cls"] in ("reranker_failure", "fusion_failure"),
}
print("\n=== PHASE 2: feature medians by group ===")
hdr = f"{'feature':<26}" + "".join(f"{g.split()[0]:>10}" for g in groups) + f"{'A-B sep':>10}"
print(hdr)
seps = []
for f in avail:
    med, vals = {}, {}
    for g, sel in groups.items():
        v = [feat[q][f] for q in feat if sel(feat[q])]
        vals[g] = v
        med[g] = statistics.median(v)
    a = vals["A generation_failure"]
    b = vals["B healthy(ok+verbose)"]
    pooled = statistics.pstdev(a + b) or 1e-9
    sep = (statistics.mean(a) - statistics.mean(b)) / pooled          # Cohen-d-like
    seps.append((abs(sep), f, sep))
    print(f"{f:<26}" + "".join(f"{med[g]:>10.3f}" for g in groups) + f"{sep:>10.2f}")

print("\n=== top 5 features by |A vs B| standardised separation ===")
for s, f, sep in sorted(seps, reverse=True)[:5]:
    print(f"  {f:<26} {sep:+.2f}")

# ---------- PHASE 3: oracles ----------
n = len(feat)
base = statistics.mean(e1[q]["base_meteor"] for q in feat)
print(f"\n=== PHASE 3: oracle upper bounds (baseline {base:.4f}, SE 0.028) ===")
cls_gain = sum(feat[q]["_delta"] for q in feat if feat[q]["_cls"] == "generation_failure")
print(f"  class oracle (route all 25 generation_failure): {cls_gain/n:+.4f}")
abs_gain = sum(max(0.0, feat[q]["_delta"]) for q in feat)
print(f"  absolute oracle (per-question max, ceiling for ANY gate): {abs_gain/n:+.4f}")
print(f"  questions where narrowing helps: {sum(feat[q]['_e1_helps'] for q in feat)}/{n}")

# ---------- PHASE 4: threshold search ----------
print("\n=== PHASE 4: threshold search (quantiles 10/25/50/75/90, both directions) ===")
print(f"  {'feature':<26}{'dir':>4}{'thr':>9}{'n_gated':>8}{'genfail':>8}"
      f"{'healthy':>8}{'prec':>7}{'est_delta':>11}")
results = []
for f in avail:
    xs = sorted(feat[q][f] for q in feat)
    for qt in (0.10, 0.25, 0.50, 0.75, 0.90):
        thr = xs[max(0, min(len(xs) - 1, int(qt * len(xs))))]
        for direction in ("<", ">"):
            gated = [q for q in feat
                     if (feat[q][f] < thr if direction == "<" else feat[q][f] > thr)]
            if not (10 <= len(gated) <= 120):
                continue
            delta = sum(feat[q]["_delta"] for q in gated) / n
            gf = sum(1 for q in gated if feat[q]["_cls"] == "generation_failure")
            hl = sum(1 for q in gated
                     if feat[q]["_cls"] in ("ok", "verbosity_failure"))
            helps = sum(1 for q in gated if feat[q]["_e1_helps"])
            results.append((delta, f, direction, thr, len(gated), gf, hl,
                            helps / len(gated)))
results.sort(reverse=True)
for d, f, dr, thr, ng, gf, hl, prec in results[:12]:
    print(f"  {f:<26}{dr:>4}{thr:>9.3f}{ng:>8}{gf:>8}{hl:>8}{prec:>7.2f}{d:>+11.4f}")
if not results:
    print("  no threshold produced a gate in the 10-120 question range")

print(f"\ntested {len(results)} feature/threshold/direction combinations - "
      f"the best is selected on the same 200 questions it is scored on, so it "
      f"is an optimistic in-sample figure, not a held-out estimate")

out = {"features_available": avail, "features_not_available": [
           "reranker_top1_score", "reranker_top1_top2_margin",
           "reranker_score_distribution", "rrf_top1_score",
           "rrf_top1_top2_margin"],
       "reason_not_available": "the retrieval trace stores child_ids, returned "
                               "counts and latency only; no stage records a score",
       "baseline_meteor": round(base, 4), "se_dev200": 0.028,
       "oracle_class_route_25": round(cls_gain / n, 4),
       "oracle_absolute_per_question_max": round(abs_gain / n, 4),
       "questions_helped_by_narrowing": sum(feat[q]["_e1_helps"] for q in feat),
       "top_features_by_separation": [
           {"feature": f, "standardised_separation": round(sep, 3)}
           for _, f, sep in sorted(seps, reverse=True)[:5]],
       "best_gates": [
           {"feature": f, "direction": dr, "threshold": round(thr, 4),
            "n_gated": ng, "generation_failure_caught": gf,
            "healthy_caught": hl, "precision_helps": round(prec, 3),
            "estimated_meteor_delta": round(d, 4)}
           for d, f, dr, thr, ng, gf, hl, prec in results[:10]],
       "per_question": {q: {k: v for k, v in feat[q].items()} for q in feat}}
Path("data/evaluation/e4_feature_table.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote data/evaluation/e4_feature_table.json")
