"""Quantify the 'right document, wrong article' pattern across every class.

Uses the benchmark's own citation regexes on the generated answer, so what
counts as a citation is identical to what counts as gold.
"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.retrieval_benchmark import ARTICLE, DOCUMENT

rows = [json.loads(l) for l in
        Path("data/evaluation/step9_failure_breakdown.jsonl").open(encoding="utf-8")]
sub = json.loads(Path("data/outputs/dev200-enriched-k20-ckpt350/submission.json")
                 .read_text(encoding="utf-8"))

for r in rows:
    pred = sub.get(r["question_id"], {}).get("answer", "")
    pa, pd = set(ARTICLE.findall(pred)), set(DOCUMENT.findall(pred))
    ga, gd = set(r["gold_articles"]), set(r["gold_documents"])
    r["pred_articles"], r["pred_documents"] = sorted(pa), sorted(pd)
    r["article_hit"] = bool(ga & pa) if ga else None
    r["document_hit"] = bool(gd & pd) if gd else None
    # The signature failure: the right law is named but a different article is quoted.
    r["right_doc_wrong_article"] = bool(gd and (gd & pd) and ga and not (ga & pa))

def pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{100*sum(xs)/len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "n/a"

print(f"{'class':<22}{'n':>4}{'artHit':>14}{'docHit':>14}{'rightDocWrongArt':>18}{'ROUGE-L':>9}")
for c in ["generation_failure", "verbosity_failure", "ok", "reranker_failure",
          "fusion_failure", "both_miss"]:
    xs = [r for r in rows if r["cls"] == c]
    if not xs: continue
    rl = statistics.mean([r["rouge_l"] for r in xs if r["rouge_l"] is not None])
    print(f"{c:<22}{len(xs):>4}{pct([r['article_hit'] for r in xs]):>14}"
          f"{pct([r['document_hit'] for r in xs]):>14}"
          f"{pct([r['right_doc_wrong_article'] for r in xs]):>18}{rl:>9.4f}")

gf = [r for r in rows if r["cls"] == "generation_failure"]
print("\ngeneration_failure detail:")
for r in sorted(gf, key=lambda x: x["meteor"])[:10]:
    print(f"  {r['question_id']:<8} M={r['meteor']:.3f} rank={r['stages']['rrf_reranker']['first_gold_rank']}"
          f" gold_art={r['gold_articles']} pred_art={r['pred_articles'][:4]}"
          f" gold_doc={r['gold_documents']} pred_doc={r['pred_documents'][:2]}")
