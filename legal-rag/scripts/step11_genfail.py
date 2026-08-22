"""Forensic analysis of the 25 generation_failure questions. ANALYSIS ONLY.

No generation, no LLM, no training, no production write. Everything is read from
the frozen winner run (ckpt-350 + enriched + k20 + floor500, METEOR 0.4880) and
from the read-only metadata DB.

Context is reconstructed from `evidence_ids` in the run's own partial.jsonl -
those are the parent segments the context builder actually placed in the prompt,
in prompt order. That is a record of what happened, not a re-derivation.
"""
import json
import re
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(".")
RUN = ROOT / "data/outputs/dev200-enriched-k20-ckpt350/partial.jsonl"
BRK = ROOT / "data/evaluation/step9_failure_breakdown.jsonl"
TRAIN = ROOT / "data/train/train.json"
SQLITE = ROOT / "storage/indexes/v1/metadata/legal.sqlite"

# Same regexes the retrieval benchmark uses, so "gold" means the same thing here
# as it does in every earlier artifact.
DOCUMENT = re.compile(r"(\d+/\d{4}/[A-ZĐ][A-ZĐ-]*|\d+-\d{4}-[A-ZĐ][A-ZĐ-]*)")
ARTICLE = re.compile(r"Điều\s+(\d+)")
KHOAN = re.compile(r"[Kk]hoản\s+(\d+)")

run = {str(json.loads(l)["question_id"]): json.loads(l) for l in RUN.open(encoding="utf-8")}
brk = {str(json.loads(l)["question_id"]): json.loads(l) for l in BRK.open(encoding="utf-8")}
train = json.loads(TRAIN.read_text(encoding="utf-8"))
N = len(run)

# ---- resolve every evidence parent once, read-only
need = sorted({e for r in run.values() for e in r["evidence_ids"]})
con = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
eva = {}
for i in range(0, len(need), 900):
    ch = need[i:i + 900]
    q = f"""SELECT parent_id, document_name, article, text, token_count
            FROM parent_chunks WHERE parent_id IN ({','.join('?' * len(ch))})"""
    for pid, dn, art, txt, tok in con.execute(q, ch):
        eva[pid] = {"document_name": dn or "", "article": art or "",
                    "text": txt or "", "token_count": tok or 0}
con.close()
missing = [e for e in need if e not in eva]
print(f"evidence parents resolved {len(eva)}/{len(need)}"
      + (f"  UNRESOLVED {len(missing)}" if missing else ""))


def hits(item, gold_docs, gold_arts):
    """Same matching rule as scripts/retrieval_benchmark.hits()."""
    hay = f"{item['document_name']}\n{item['article']}\n{item['text']}"
    return (gold_docs & set(DOCUMENT.findall(hay)),
            gold_arts & set(ARTICLE.findall(hay)))


rows = []
for qid, r in run.items():
    b = brk[qid]
    gold_docs, gold_arts = set(b["gold_documents"]), set(b["gold_articles"])
    ans = r["answer"] or ""
    ref = train[qid]["answer"]
    ev = r["evidence_ids"]

    pos, gold_ev = None, []
    for i, e in enumerate(ev, start=1):
        item = eva.get(e)
        if not item:
            continue
        d, a = hits(item, gold_docs, gold_arts)
        if d or a:
            gold_ev.append(e)
            if pos is None:
                pos = i

    cited = set(ARTICLE.findall(ans))
    ctx_articles = {m for e in ev if e in eva
                    for m in ARTICLE.findall(eva[e]["article"])}
    # Which clauses the expert answer leans on, and whether the gold article's
    # text in context actually contains them.
    ref_clauses = set(KHOAN.findall(ref))
    gold_text = "\n".join(eva[e]["text"] for e in gold_ev if e in eva)
    ctx_clauses = set(KHOAN.findall(gold_text))
    missing_clauses = sorted(ref_clauses - ctx_clauses, key=int)

    rows.append({
        "question_id": qid,
        "cls": b["cls"],
        "gold_article": sorted(gold_arts, key=int),
        "gold_document": sorted(gold_docs),
        "gold_in_context": bool(gold_ev),
        "gold_position_in_context": pos,
        "n_evidence_in_context": len(ev),
        "context_token_count": sum(eva[e]["token_count"] for e in ev if e in eva),
        "answer_words": len(ans.split()),
        "reference_words": len(ref.split()),
        "article_hit": bool(gold_arts & cited),
        "cited_articles": sorted(cited, key=int),
        "cited_not_in_context": sorted(cited - ctx_articles, key=int),
        "neighbour_cited": sorted(
            {c for c in cited for g in gold_arts if 0 < abs(int(c) - int(g)) <= 2},
            key=int),
        "ref_clauses": sorted(ref_clauses, key=int),
        "missing_clauses_in_context": missing_clauses,
        # Both of these are constant across all 200 rows - see the report.
        "grounded": r["grounded"],
        "citations": r["citations"],
        "meteor": b["meteor"],
        "rouge_l": b["rouge_l"],
    })
by_id = {r["question_id"]: r for r in rows}

# ---- sanity: are the run's internal validator flags informative at all?
print(f"\ngrounded=True on {sum(1 for r in rows if r['grounded'])}/{N} questions; "
      f"non-empty citations on {sum(1 for r in rows if r['citations'])}/{N}")
ok_rows = [r for r in rows if r["cls"] == "ok"]
print(f"  among the {len(ok_rows)} healthy 'ok' questions (mean METEOR "
      f"{statistics.mean(x['meteor'] for x in ok_rows):.4f}): grounded=True on "
      f"{sum(1 for r in ok_rows if r['grounded'])}, citations on "
      f"{sum(1 for r in ok_rows if r['citations'])}")


def classify(r):
    """Priority order matters; first rule that fires wins. Every rule keys off a
    field measured above, never off a reading of the answer text."""
    if not r["gold_article"]:
        return ("NO_PARSEABLE_GOLD_ARTICLE",
                "reference cites no article number, so article_hit is False by "
                "construction and no citation-based class can be assigned")
    if not r["gold_in_context"]:
        return "GOLD_NOT_IN_CONTEXT", "gold article absent from the context that was built"
    if not r["article_hit"] and r["neighbour_cited"]:
        return ("CONTEXT_ORDER_OR_NEIGHBOR_NOISE",
                f"cited neighbour article {r['neighbour_cited']} instead of gold "
                f"{r['gold_article']}, gold sat at context position "
                f"{r['gold_position_in_context']}/{r['n_evidence_in_context']}")
    if not r["article_hit"]:
        return ("GOLD_IN_CONTEXT_BUT_WRONG",
                f"gold at context position {r['gold_position_in_context']} but answer "
                f"cited {r['cited_articles'] or 'no article'}")
    if r["missing_clauses_in_context"]:
        return ("GOLD_CONTEXT_INSUFFICIENT",
                f"right article cited, but clause(s) {r['missing_clauses_in_context']} "
                f"used by the reference are not in the retrieved text")
    if r["cited_not_in_context"]:
        return ("CITATION_OR_GROUNDING_FAILURE",
                f"cited article(s) {r['cited_not_in_context']} that are not in context")
    if r["answer_words"] < 50:
        return ("PROMPT_FOLLOWING_FAILURE",
                f"answer only {r['answer_words']} words against a "
                f"{r['reference_words']}-word reference")
    return ("TRUE_GENERATION_FAILURE",
            "gold in context, gold article cited, clauses present - content still wrong")


for r in rows:
    r["e6_class"], r["failure_reason"] = classify(r)

gf = [r for r in rows if r["cls"] == "generation_failure"]
print(f"\n{'='*100}\nPER-QUESTION: {len(gf)} generation_failure\n{'='*100}")
hdr = (f"{'qid':<8}{'gold':>6}{'inCtx':>7}{'pos':>5}{'ctxTok':>8}{'ansW':>6}"
       f"{'refW':>6}{'artHit':>8}{'citOK':>7}  class")
print(hdr)
for r in sorted(gf, key=lambda x: x["e6_class"]):
    cit = "no" if r["cited_not_in_context"] or not r["cited_articles"] else "yes"
    print(f"{r['question_id']:<8}{(r['gold_article'] or ['-'])[0]:>6}"
          f"{'yes' if r['gold_in_context'] else 'NO':>7}"
          f"{str(r['gold_position_in_context'] or '-'):>5}"
          f"{r['context_token_count']:>8}{r['answer_words']:>6}{r['reference_words']:>6}"
          f"{'yes' if r['article_hit'] else 'no':>8}{cit:>7}  {r['e6_class']}")

print(f"\n{'-'*100}\nfailure_reason per question\n{'-'*100}")
for r in sorted(gf, key=lambda x: x["e6_class"]):
    print(f"  {r['question_id']:<8} {r['e6_class']}")
    print(f"           {r['failure_reason']}")

# ---- class table
print(f"\n{'='*100}\nCLASS SUMMARY (25 generation_failure)\n{'='*100}")
print(f"{'class':<36}{'n':>4}{'%':>7}{'meanMETEOR':>12}{'medWords':>10}"
      f"{'medRefW':>9}{'goldInCtx%':>12}")
counts = Counter(r["e6_class"] for r in gf)
for cls, n in counts.most_common():
    g = [r for r in gf if r["e6_class"] == cls]
    print(f"{cls:<36}{n:>4}{100*n/len(gf):>6.1f}%"
          f"{statistics.mean(r['meteor'] for r in g):>12.4f}"
          f"{statistics.median(r['answer_words'] for r in g):>10.0f}"
          f"{statistics.median(r['reference_words'] for r in g):>9.0f}"
          f"{100*sum(r['gold_in_context'] for r in g)/n:>11.0f}%")

# ---- recoverable
print(f"\n{'='*100}\nRECOVERABLE generation failures\n{'='*100}")
recoverable = [r for r in gf if r["gold_in_context"]
               and not r["missing_clauses_in_context"]]
print("definition: gold article present in the built context AND every clause the")
print("reference leans on is present in that text - retrieval and evidence both")
print("did their job, so the loss is downstream of them.")
print(f"  recoverable: {len(recoverable)}/{len(gf)}")
print(f"  blocked by missing evidence: "
      f"{sum(1 for r in gf if r['gold_in_context'] and r['missing_clauses_in_context'])}")
print(f"  blocked by missing gold in context: {sum(1 for r in gf if not r['gold_in_context'])}")

healthy = statistics.mean(r["meteor"] for r in rows if r["cls"] == "ok")
print(f"\nceiling arithmetic, priced at the healthy 'ok' mean {healthy:.4f}")
print("(generous - these are the hardest questions - so each figure is a ceiling)")
for label, group in (("all 25 generation_failure", gf),
                     ("recoverable subset", recoverable),
                     ("neighbour-noise subset",
                      [r for r in gf if r["e6_class"] == "CONTEXT_ORDER_OR_NEIGHBOR_NOISE"]),
                     ("insufficient-evidence subset",
                      [r for r in gf if r["e6_class"] == "GOLD_CONTEXT_INSUFFICIENT"])):
    gain = sum(max(0.0, healthy - r["meteor"]) for r in group) / N
    print(f"  {label:<32} n={len(group):>3}  +{gain:.4f}  ({gain/0.028:.2f}x SE)")

# ---- context position: does burying gold correlate with failure?
print(f"\n{'='*100}\nCONTEXT POSITION of gold, generation_failure vs healthy\n{'='*100}")
for label, group in (("generation_failure", gf),
                     ("ok (healthy)", [r for r in rows if r["cls"] == "ok"]),
                     ("verbosity_failure", [r for r in rows if r["cls"] == "verbosity_failure"])):
    pos = [r["gold_position_in_context"] for r in group if r["gold_position_in_context"]]
    tok = [r["context_token_count"] for r in group]
    if pos:
        print(f"  {label:<20} n={len(group):>3}  gold pos median {statistics.median(pos):>5.1f}"
              f"  mean {statistics.mean(pos):>5.2f}  max {max(pos):>3}"
              f"   ctx tokens median {statistics.median(tok):>6.0f}")

out = ROOT / "data/evaluation/step11_generation_failure_forensics.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
               encoding="utf-8")
print(f"\nwrote {out}  ({len(rows)} rows, all classes)")
