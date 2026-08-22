"""PHASE 3 - compare E1 (CONTEXT_NEIGHBOR_WINDOW=0) against the frozen baseline.

Per-question, on the same 200 dev questions, with the 25 generation_failure
questions called out separately. Citation hits use the benchmark's own regexes,
so what counts as a citation is identical to what defined gold.
"""
import json, statistics, sys
from pathlib import Path

sys.path.insert(0, ".")
from app.evaluation.generation_metrics import meteor, rouge_l
from scripts.retrieval_benchmark import ARTICLE, DOCUMENT

BASE = Path("data/outputs/dev200-enriched-k20-ckpt350/submission.json")
E1 = Path("data/outputs/dev200-e1-nbwindow0/submission.json")
CLS = Path("data/evaluation/step8_per_question.jsonl")
BRK = Path("data/evaluation/step9_failure_breakdown.jsonl")

base = json.loads(BASE.read_text(encoding="utf-8"))
e1 = json.loads(E1.read_text(encoding="utf-8"))
train = json.loads(Path("data/train/train.json").read_text(encoding="utf-8"))
cls = {json.loads(l)["question_id"]: json.loads(l) for l in CLS.open(encoding="utf-8")}
brk = {json.loads(l)["question_id"]: json.loads(l) for l in BRK.open(encoding="utf-8")}

assert set(base) == set(e1), "the two runs cover different question sets"

def cite(text):
    return set(ARTICLE.findall(text)), set(DOCUMENT.findall(text))

rows = []
for qid in base:
    ref = train[qid]["answer"]
    b, e = base[qid]["answer"], e1[qid]["answer"]
    ba, bd = cite(b)
    ea, ed = cite(e)
    ga = set(brk[qid]["gold_articles"])
    gd = set(brk[qid]["gold_documents"])
    rows.append({
        "question_id": qid,
        "cls": cls[qid]["cls"],
        "gold_articles": sorted(ga), "gold_documents": sorted(gd),
        "base_meteor": meteor(b, ref), "e1_meteor": meteor(e, ref),
        "base_rouge_l": rouge_l(b, ref), "e1_rouge_l": rouge_l(e, ref),
        "base_words": len(b.split()), "e1_words": len(e.split()),
        "ref_words": len(ref.split()),
        "base_article_hit": bool(ga & ba) if ga else None,
        "e1_article_hit": bool(ga & ea) if ga else None,
        "base_right_doc_wrong_article": bool(gd and (gd & bd) and ga and not (ga & ba)),
        "e1_right_doc_wrong_article": bool(gd and (gd & ed) and ga and not (ga & ea)),
        "base_articles": sorted(ba)[:5], "e1_articles": sorted(ea)[:5],
        "base_head": b[:160], "e1_head": e[:160],
    })
for r in rows:
    r["delta_meteor"] = r["e1_meteor"] - r["base_meteor"]
    r["delta_rouge_l"] = r["e1_rouge_l"] - r["base_rouge_l"]

Path("data/evaluation/e1_per_question.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

def block(title, xs):
    if not xs:
        return
    bm = statistics.mean(r["base_meteor"] for r in xs)
    em = statistics.mean(r["e1_meteor"] for r in xs)
    br = statistics.mean(r["base_rouge_l"] for r in xs)
    er = statistics.mean(r["e1_rouge_l"] for r in xs)
    up = sum(1 for r in xs if r["delta_meteor"] > 0.02)
    down = sum(1 for r in xs if r["delta_meteor"] < -0.02)
    same = len(xs) - up - down
    def hit(k):
        v = [r[k] for r in xs if r[k] is not None]
        return f"{100*sum(v)/len(v):.0f}% ({sum(v)}/{len(v)})" if v else "n/a"
    print(f"\n=== {title}  n={len(xs)} ===")
    print(f"  METEOR   base {bm:.4f} -> E1 {em:.4f}   delta {em-bm:+.4f}")
    print(f"  ROUGE-L  base {br:.4f} -> E1 {er:.4f}   delta {er-br:+.4f}")
    print(f"  words    base {statistics.median(r['base_words'] for r in xs):.0f}"
          f" -> E1 {statistics.median(r['e1_words'] for r in xs):.0f}"
          f"   (ref {statistics.median(r['ref_words'] for r in xs):.0f})")
    print(f"  article hit  base {hit('base_article_hit')} -> E1 {hit('e1_article_hit')}")
    print(f"  right doc wrong article  base "
          f"{sum(r['base_right_doc_wrong_article'] for r in xs)}"
          f" -> E1 {sum(r['e1_right_doc_wrong_article'] for r in xs)}")
    print(f"  rescued(>+0.02) {up}   regressed(<-0.02) {down}   unchanged {same}")

block("ALL", rows)
for c in ["generation_failure", "reranker_failure", "fusion_failure",
          "both_miss", "verbosity_failure", "ok"]:
    block(c, [r for r in rows if r["cls"] == c])

gf = [r for r in rows if r["cls"] == "generation_failure"]
print("\ngeneration_failure, per question:")
print(f"  {'qid':<9}{'base':>7}{'E1':>8}{'delta':>9}  {'gold':<12}{'baseArt':<14}{'e1Art':<14}")
for r in sorted(gf, key=lambda x: -x["delta_meteor"]):
    print(f"  {r['question_id']:<9}{r['base_meteor']:>7.3f}{r['e1_meteor']:>8.3f}"
          f"{r['delta_meteor']:>+9.3f}  {str(r['gold_articles']):<12}"
          f"{str(r['base_articles'][:3]):<14}{str(r['e1_articles'][:3]):<14}")
