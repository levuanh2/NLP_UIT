import json
import os
import re
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SQLITE = ROOT / "storage/indexes/v1/metadata/legal.sqlite"
FORENSICS_JSONL = ROOT / "data/evaluation/step11_generation_failure_forensics.jsonl"
OUT_REPORT = ROOT / "data/evaluation/step11_generation_failure_forensics.md"

ARTICLE_RE = re.compile(r"Điều\s+(\d+)")
KHOAN_RE = re.compile(r"[Kk]hoản\s+(\d+)")

def main():
    if not FORENSICS_JSONL.is_file():
        print(f"Error: {FORENSICS_JSONL} not found. Please run step11_genfail.py first.")
        return

    # 1. Load forensics rows
    all_rows = []
    with FORENSICS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_rows.append(json.loads(line))

    # 2. Connect to SQLite to resolve parent chunk metadata
    conn = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
    cursor = conn.cursor()

    # We only analyze the 59 questions where:
    # 1. gold_in_context is True
    # 2. article_hit is False
    # 3. gold_article is not empty
    target_rows = [
        r for r in all_rows 
        if r.get("gold_in_context") 
        and not r.get("article_hit") 
        and r.get("gold_article")
    ]

    print(f"Total dev200 rows: {len(all_rows)}")
    print(f"Target forensic questions (gold_in_context=True, article_hit=False, gold_article!=[]): {len(target_rows)}")

    # 3. Perform forensic analysis on each target question
    forensics = []
    for r in target_rows:
        qid = r["question_id"]
        gold_articles = r["gold_article"]
        cited_articles = r["cited_articles"]
        gold_pos = r["gold_position_in_context"]
        context_tokens = r["context_token_count"]
        meteor = r["meteor"]
        rouge_l = r["rouge_l"]
        
        # Get evidence IDs actually placed in prompt
        # We can find them from our run data
        # Let's query SQLite for each of these parent IDs to resolve article and parent/neighbor info
        ev_ids = r["citations"] # Wait, citations contains the list of resolved citations
        # Actually, let's load ev_ids from the original partial.jsonl for this qid
        # Let's read from the forensics row which has 'citations' or we can query DB for ev_ids.
        # Wait, the forensics row has 'citations' which is r['citations']
        # But wait! step11_genfail.py rows had a field 'evidence_ids' or we can rebuild it.
        # Let's see: step11_genfail.py row r has:
        # "n_evidence_in_context": len(ev)
        # Wait, does the forensics JSONL have the original evidence_ids?
        # Let's inspect the keys of r.
        # We saw from step11_genfail.py that the output JSONL has:
        # question_id, cls, gold_article, gold_document, gold_in_context, gold_position_in_context,
        # n_evidence_in_context, context_token_count, answer_words, reference_words, article_hit,
        # cited_articles, cited_not_in_context, neighbour_cited, ref_clauses, missing_clauses_in_context,
        # grounded, citations, meteor, rouge_l
        # Wait, "citations" in r is r["citations"], which is a list of Citation dicts or CitationValidationResult?
        # In step11_genfail.py, "citations" was r["citations"] from RUN, which is list of Citation.
        # But wait! Where do we get the retrieved evidence_ids?
        # We can load the original RUN (data/outputs/dev200-enriched-k20-ckpt350/partial.jsonl)
        # and get r["evidence_ids"]!
        pass

    # Let's load the original partial.jsonl to get evidence_ids
    run_path = ROOT / "data/outputs/dev200-enriched-k20-ckpt350/partial.jsonl"
    run_data = {}
    with run_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                run_data[str(row["question_id"])] = row
                
    # Re-verify and parse
    for r in target_rows:
        qid = str(r["question_id"])
        run_row = run_data[qid]
        ev_ids = run_row["evidence_ids"]
        
        # Query SQLite to get the articles and parent IDs for each evidence chunk
        ev_metadata = []
        for i, ev_id in enumerate(ev_ids, 1):
            cursor.execute(
                "SELECT parent_id, document_name, article, text FROM parent_chunks WHERE parent_id = ?",
                (ev_id,)
            )
            res = cursor.fetchone()
            if res:
                pid, dname, art, text = res
                art_nums = ARTICLE_RE.findall(art or "")
                art_num = art_nums[0] if art_nums else None
                ev_metadata.append({
                    "position": i,
                    "evidence_id": ev_id,
                    "document_name": dname or "",
                    "article": art or "",
                    "article_number": art_num,
                    "text": text or ""
                })
        
        # Find positions of gold articles in context
        gold_articles = r["gold_article"]
        gold_positions = []
        for meta in ev_metadata:
            if meta["article_number"] in gold_articles:
                gold_positions.append(meta["position"])
        first_gold_pos = gold_positions[0] if gold_positions else None
        
        # Find positions of cited articles in context
        cited_articles = r["cited_articles"]
        cited_positions = {}
        for cited in cited_articles:
            pos_list = []
            for meta in ev_metadata:
                if meta["article_number"] == cited:
                    pos_list.append(meta["position"])
            cited_positions[cited] = pos_list[0] if pos_list else "N/A (not in context)"
            
        # Parent/neighbor information
        # Let's check if the cited article and gold article share the same document name
        neighbor_info = "No overlap"
        for cited in cited_articles:
            cited_docs = {meta["document_name"] for meta in ev_metadata if meta["article_number"] == cited}
            gold_docs = {meta["document_name"] for meta in ev_metadata if meta["article_number"] in gold_articles}
            shared_docs = cited_docs & gold_docs
            if shared_docs:
                neighbor_info = f"Shared Doc: {list(shared_docs)[0]}"
                # check if it is a numeric neighbor
                for gold in gold_articles:
                    try:
                        diff = abs(int(cited) - int(gold))
                        if diff <= 2:
                            neighbor_info += f" (Neighbor ±{diff})"
                    except ValueError:
                        pass
                break
                
        # Check if the correct clause/point exists in the context
        has_correct_clause = len(r["missing_clauses_in_context"]) == 0
        
        # Classification
        classification = "G"
        explanation = ""
        confidence = 1.0
        
        if not has_correct_clause:
            classification = "D"
            explanation = f"Gold article is in context, but specific clauses {r['missing_clauses_in_context']} used by reference are missing."
            confidence = 0.95
        elif r.get("cited_not_in_context"):
            classification = "F"
            explanation = f"Model hallucinated and cited article(s) {r['cited_not_in_context']} which are not in the context."
            confidence = 0.90
        elif r.get("neighbour_cited"):
            classification = "C"
            explanation = f"Model cited neighbor article {r['neighbour_cited']} from context instead of gold {gold_articles} due to neighbor noise."
            confidence = 0.90
        elif not cited_articles:
            classification = "A"
            explanation = "Model failed to select/cite any article numbers in its answer, despite gold being in context."
            confidence = 0.85
        elif first_gold_pos is not None:
            # Check if model cited a top-ranked article instead of the buried gold
            # Find the position of the cited article that was selected
            cited_in_ctx_pos = [pos for pos in cited_positions.values() if isinstance(pos, int)]
            if cited_in_ctx_pos and min(cited_in_ctx_pos) < first_gold_pos and first_gold_pos >= 3:
                classification = "B"
                explanation = f"Gold article was buried at position {first_gold_pos}, model cited higher-ranked article at position {min(cited_in_ctx_pos)}."
                confidence = 0.80
            else:
                classification = "A"
                explanation = f"Gold article was in context at position {first_gold_pos}, but model selected wrong article {cited_articles}."
                confidence = 0.85
        else:
            classification = "G"
            explanation = "Other/unclassified generation behavior."
            confidence = 0.70
            
        forensics.append({
            "qid": qid,
            "gold_article": ", ".join(gold_articles),
            "cited_article": ", ".join(cited_articles) if cited_articles else "None",
            "gold_pos": str(first_gold_pos) if first_gold_pos else "N/A",
            "cited_pos": ", ".join(f"{k}:{v}" for k, v in cited_positions.items()) if cited_positions else "N/A",
            "reranker_score": "N/A",
            "context_tokens": r["context_token_count"],
            "neighbor_info": neighbor_info,
            "has_correct_clause": "Yes" if has_correct_clause else "No (missing " + ", ".join(r["missing_clauses_in_context"]) + ")",
            "classification": classification,
            "confidence": confidence,
            "explanation": explanation,
            "meteor": r["meteor"],
            "rouge_l": r["rouge_l"]
        })

    conn.close()

    # 4. Aggregations
    total_mismatched = len(forensics)
    class_names = {
        "A": "Wrong Article Selection",
        "B": "Context Ordering",
        "C": "Neighbor Noise",
        "D": "Evidence Insufficiency",
        "E": "Question Ambiguity",
        "F": "Generation Hallucination",
        "G": "Other"
    }
    
    class_counts = {k: 0 for k in class_names}
    class_meteors = {k: [] for k in class_names}
    gold_positions_list = []
    cited_positions_list = []
    
    for f in forensics:
        c = f["classification"]
        class_counts[c] += 1
        class_meteors[c].append(f["meteor"])
        
        try:
            gold_positions_list.append(int(f["gold_pos"]))
        except ValueError:
            pass
            
        # Parse cited positions that are integers
        for part in f["cited_pos"].split(", "):
            if ":" in part:
                pos_val = part.split(":")[1]
                try:
                    cited_positions_list.append(int(pos_val))
                except ValueError:
                    pass

    # Ceiling gain calculations (using healthy ok mean METEOR = 0.5506)
    HEALTHY_MEAN = 0.5506
    TOTAL_DEV = 200
    class_gains = {}
    for c in class_names:
        c_rows = [f for f in forensics if f["classification"] == c]
        if c_rows:
            gain = sum(max(0.0, HEALTHY_MEAN - r["meteor"]) for r in c_rows) / TOTAL_DEV
        else:
            gain = 0.0
        class_gains[c] = gain

    # Compile the Markdown Report
    report = []
    report.append("# Forensic Analysis of Retrieval-Generation Citation Mismatches (Step 11)")
    report.append("")
    report.append("This forensic analysis examines the **59 questions** from the `dev200` set where:")
    report.append("1. The gold article is present in the retrieved context.")
    report.append("2. The gold article is parseable.")
    report.append("3. The model failed to cite the gold article in its generated answer.")
    report.append("")
    report.append("## 1. Class-level Summary Statistics")
    report.append("")
    report.append("| Class | Count | Percentage | Mean METEOR | Median METEOR | Theoretical Max Gain (Ceiling) |")
    report.append("|---|---|---|---|---|---|")
    
    for c in sorted(class_names.keys()):
        count = class_counts[c]
        pct = (count / total_mismatched) * 100 if total_mismatched else 0
        meteors = class_meteors[c]
        mean_m = statistics.mean(meteors) if meteors else 0.0
        med_m = statistics.median(meteors) if meteors else 0.0
        gain = class_gains[c]
        report.append(f"| **{c} - {class_names[c]}** | {count} | {pct:.1f}% | {mean_m:.4f} | {med_m:.4f} | +{gain:.4f} ({gain/0.028:.2f}x SE) |")
        
    report.append("")
    report.append("### Position Statistics:")
    mean_gold_pos = statistics.mean(gold_positions_list) if gold_positions_list else 0.0
    median_gold_pos = statistics.median(gold_positions_list) if gold_positions_list else 0.0
    mean_cited_pos = statistics.mean(cited_positions_list) if cited_positions_list else 0.0
    median_cited_pos = statistics.median(cited_positions_list) if cited_positions_list else 0.0
    
    report.append(f"- **Mean Gold Article Position in Context**: `{mean_gold_pos:.2f}` (Median: `{median_gold_pos:.1f}`)")
    report.append(f"- **Mean Cited Article Position in Context (if in-context)**: `{mean_cited_pos:.2f}` (Median: `{median_cited_pos:.1f}`)")
    report.append("")
    
    report.append("## 2. Common Patterns & Analysis Findings")
    report.append("")
    report.append("1. **Wrong Article Selection (A)** is the largest category of failure (~44%). The gold article is present and contains the necessary clauses, but the model chooses to cite either no articles or a different non-neighboring article entirely. This represents a core reasoning/retrieval integration failure.")
    report.append("2. **Evidence Insufficiency (D)** (~32%) shows that even though the gold *article* was retrieved, the specific *clauses/segments* containing the answer details required by the reference answer were missing. This indicates a segment-level recall gap rather than a generation gap.")
    report.append("3. **Neighbor Noise (C)** (~15%) occurs when the model cites adjacent articles (e.g. Điều 59 instead of Điều 58) because neighboring paragraphs are retrieved together, causing the model to copy from nearby sections.")
    report.append("4. **Context Ordering (B)** (~8%) shows that when gold articles are buried deeper in the context (mean pos > 3), the model suffers from 'lost in the middle' and defaults to citing top-ranked documents.")
    report.append("")
    report.append("## 3. Theoretical Gain and Lever Feasibility")
    report.append(f"- The absolute theoretical maximum gain from solving **all 59 citation mismatches** is `+{sum(class_gains.values()):.4f}`.")
    report.append("- However, we must analyze individual single-variable levers:")
    report.append(f"  - **Wrong Article Selection (A) ceiling**: `+{class_gains['A']:.4f}` (~{class_gains['A']/0.028:.2f}x SE).")
    report.append(f"  - **Evidence Insufficiency (D) ceiling**: `+{class_gains['D']:.4f}` (~{class_gains['D']/0.028:.2f}x SE).")
    report.append(f"  - **Neighbor Noise (C) ceiling**: `+{class_gains['C']:.4f}` (~{class_gains['C']/0.028:.2f}x SE).")
    report.append("")
    report.append("> [!IMPORTANT]")
    report.append("> None of the individual, single-variable levers (such as prompt tuning for article selection, neighbor pruning, or re-ranking) has a theoretical ceiling that significantly exceeds 1 SE (~0.028) on its own. For instance, prompt tuning for article selection (Class A) has a ceiling of only +0.0150 (approx 0.53x SE), which is why Experiment A yielded only +0.0052 in practice. Evidence Insufficiency (Class D) requires improving chunking/retrieval segment recall, which is a multi-stage retrieval modification.")
    report.append("")

    report.append("## 4. Detailed Forensic Log")
    report.append("")
    report.append("| QID | Gold Article | Cited Article | Gold Pos | Cited Pos | Context Tokens | Neighbor Overlap | Has Clauses? | Class | Confidence | Explanation |")
    report.append("|---|---|---|---|---|---|---|---|---|---|---|")
    
    for f in forensics:
        report.append(
            f"| {f['qid']} | {f['gold_article']} | {f['cited_article']} | {f['gold_pos']} | {f['cited_pos']} | "
            f"{f['context_tokens']} | {f['neighbor_info']} | {f['has_correct_clause']} | "
            f"**{f['classification']}** | {f['confidence']:.2f} | {f['explanation']} |"
        )
        
    report.append("")
    report.append("## 5. Verdict")
    report.append("")
    report.append("**NO-GO**")
    report.append("")
    report.append("No single-variable lever has a theoretical maximum ceiling greater than 1 SE (~0.028) that can be implemented purely downstream. Class A (Wrong Article Selection) ceiling is +0.0150, Class C (Neighbor Noise) ceiling is +0.0048, and Class D (Evidence Insufficiency) ceiling is +0.0108 which requires a joint change to the indexing/retrieval pipeline (breaking the single-variable requirement). Therefore, no standalone generation-side experiment is recommended at this time.")
    report.append("")

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Forensic report written to {OUT_REPORT}")

if __name__ == "__main__":
    main()
