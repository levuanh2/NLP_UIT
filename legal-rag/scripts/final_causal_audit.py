import json
import os
import re
import sqlite3
import statistics
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SQLITE = ROOT / "storage/indexes/v1/metadata/legal.sqlite"
TRACE_JSONL = ROOT / "data/evaluation/retrieval_enrichment_ab/B-enriched-index-enriched-reranker-k20/per_question.jsonl"
TRAIN_JSON = ROOT / "data/train/train.json"
PARTIAL_JSONL = ROOT / "data/outputs/dev200-enriched-k20-ckpt350/partial.jsonl"
FORENSICS_JSONL = ROOT / "data/evaluation/step11_generation_failure_forensics.jsonl"
BREAKDOWN_JSONL = ROOT / "data/evaluation/step9_failure_breakdown.jsonl"

DOCUMENT = re.compile(r"(\d+/\d{4}/[A-ZĐ][A-ZĐ-]*|\d+-\d{4}-[A-ZĐ][A-ZĐ-]*)")
ARTICLE_RE = re.compile(r"Điều\s+(\d+)")
KHOAN_RE = re.compile(r"[Kk]hoản\s+(\d+)")
DIEM_RE = re.compile(r"[Đđ]iểm\s+([a-zđ]+)")

def normalize_doc_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFD", name.casefold())
    s = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    s = s.replace("đ", "d").replace("/", "-")
    s = re.sub(r"[^a-z0-9-]", "", s)
    return s

def main():
    # Load files
    trace = {}
    with TRACE_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                trace[str(row["question_id"])] = row
                
    train = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    
    partial = {}
    with PARTIAL_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                partial[str(row["question_id"])] = row

    forensics = {}
    if FORENSICS_JSONL.is_file():
        with FORENSICS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    forensics[str(row["question_id"])] = row

    breakdown = {}
    with BREAKDOWN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                breakdown[str(row["question_id"])] = row

    conn = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Taxonomy counts
    total_q = len(trace)
    
    categories = {
        "A": [], # retriever miss gold hoàn toàn
        "B": [], # gold sống qua retrieval nhưng mất ở reranker
        "C": [], # gold sống qua reranker nhưng mất ở context builder
        "D": [], # gold đầy đủ trong context nhưng answer sai
        "E": [], # gold article có nhưng evidence không đủ khoản/điểm
        "Other": []
    }
    
    ok_count = 0

    for qid, t_row in trace.items():
        b_row = breakdown[qid]
        p_row = partial[qid]
        f_row = forensics.get(qid)
        
        ref_answer = train[qid]["answer"]
        gold_articles = set(t_row["gold_articles"])
        gold_documents = set(t_row["gold_documents"])
        ref_clauses = set(KHOAN_RE.findall(ref_answer))
        ref_points = set(DIEM_RE.findall(ref_answer))
        
        # Check if question has no parseable gold
        if not gold_articles and not gold_documents:
            categories["Other"].append(qid)
            continue
            
        if b_row["cls"] == "ok":
            ok_count += 1
            continue
            
        # Check where the gold article was lost in retrieval
        # Did dense or bm25 retrieve it?
        dense_ids = t_row["dense"]["child_ids"]
        bm25_ids = t_row["bm25"]["child_ids"]
        
        def check_art_in_list(child_ids):
            for cid in child_ids:
                cursor.execute("SELECT article, document_name FROM child_chunks WHERE child_id = ?", (cid,))
                res = cursor.fetchone()
                if res:
                    art, dname = res
                    art_nums = ARTICLE_RE.findall(art or "")
                    if any(normalize_doc_name(doc) in normalize_doc_name(dname or "") for doc in gold_documents):
                        if any(a in gold_articles for a in art_nums):
                            return True
            return False
            
        in_dense = check_art_in_list(dense_ids)
        in_bm25 = check_art_in_list(bm25_ids)
        
        in_retrieval_pool = in_dense or in_bm25
        
        if not in_retrieval_pool:
            categories["A"].append(qid)
            continue
            
        # If in pool, did it survive the Reranker?
        reranker_ids = t_row["rrf_reranker"]["child_ids"]
        in_reranker = check_art_in_list(reranker_ids)
        
        if not in_reranker:
            categories["B"].append(qid)
            continue
            
        # If in reranker, did it survive Context Builder?
        evidence_ids = p_row["evidence_ids"]
        retrieved_texts = []
        retrieved_articles = set()
        for pid in evidence_ids:
            cursor.execute("SELECT text, article, document_name FROM parent_chunks WHERE parent_id = ?", (pid,))
            res = cursor.fetchone()
            if res:
                txt, art, dname = res
                retrieved_texts.append(txt or "")
                art_nums = ARTICLE_RE.findall(art or "")
                if any(normalize_doc_name(doc) in normalize_doc_name(dname or "") for doc in gold_documents):
                    for a in art_nums:
                        retrieved_articles.add(a)
                        
        in_final_context = len(gold_articles & retrieved_articles) > 0
        
        if not in_final_context:
            categories["C"].append(qid)
            continue
            
        # If in final context, check if evidence is complete (clauses/points)
        full_context_text = "\n".join(retrieved_texts)
        stage_clauses = set(KHOAN_RE.findall(full_context_text))
        stage_points = set(DIEM_RE.findall(full_context_text))
        
        has_all_clauses = len(ref_clauses & stage_clauses) == len(ref_clauses)
        has_all_points = len(ref_points & stage_points) == len(ref_points)
        
        evidence_complete = has_all_clauses and has_all_points
        
        if not evidence_complete:
            categories["E"].append(qid)
        else:
            categories["D"].append(qid)

    print("Failure Taxonomy Counts:")
    for k, v in categories.items():
        print(f"  Category {k}: {len(v)} questions")
    print(f"  OK Class: {ok_count} questions")
    
    # Save taxonomy results to markdown file
    out_lines = [
        "# Mutually-Exclusive Causal Audit Report",
        "",
        "## 1. Taxonomy Breakdown",
        "",
        "| Category | Description | Count | Percentage |",
        "|---|---|---|---|",
        f"| **A** | Retriever miss gold hoàn toàn | {len(categories['A'])} | {len(categories['A'])/total_q*100:.1f}% |",
        f"| **B** | Gold sống qua retrieval nhưng mất ở Reranker | {len(categories['B'])} | {len(categories['B'])/total_q*100:.1f}% |",
        f"| **C** | Gold sống qua reranker nhưng mất ở Context Builder | {len(categories['C'])} | {len(categories['C'])/total_q*100:.1f}% |",
        f"| **D** | Gold đầy đủ trong context nhưng answer sai | {len(categories['D'])} | {len(categories['D'])/total_q*100:.1f}% |",
        f"| **E** | Gold article có nhưng evidence không đủ khoản/điểm | {len(categories['E'])} | {len(categories['E'])/total_q*100:.1f}% |",
        f"| **Other** | Không phân loại được hoặc không có gold parseable | {len(categories['Other'])} | {len(categories['Other'])/total_q*100:.1f}% |",
        f"| **OK** | Trả lời đúng (class ok) | {ok_count} | {ok_count/total_q*100:.1f}% |",
        "",
        "## 2. Overlap and Causal Path Analysis",
        "- **Retrieval Level Bottleneck (A + B)**: `A` (retriever miss) and `B` (reranker drop) represent the absolute trần giới hạn của retrieval. Since these documents are never retrieved or are filtered out before context building, they cannot be resolved by downstream prompt engineering or context expansion.",
        "- **Context Level Bottleneck (C + E)**: `C` (context builder drop) and `E` (incomplete clauses/points) represent context construction issues. Here, the article is hit but the exact clauses are missing.",
        "- **Generation Level Bottleneck (D)**: `D` is the pure generation-side bottleneck (wrong article selection, citation formatting error, or hallucination). This represents the ceiling of prompt-only optimizations.",
        "",
        "## 3. Ceiling Analysis for Combined Interventions",
        ""
    ]
    
    baseline_scores = {}
    for qid in trace:
        baseline_scores[qid] = breakdown[qid]["meteor"]
        
    mean_baseline = statistics.mean(baseline_scores.values())
    HEALTHY_MEAN = 0.5506
    
    # Calculate ceiling for each category
    out_lines.append("| Category | Count | Sum Baseline METEOR | Oracle METEOR Gain | Ceiling METEOR |")
    out_lines.append("|---|---|---|---|---|")
    for k, v in sorted(categories.items()):
        if not v:
            continue
        sum_base = sum(baseline_scores[q] for q in v)
        gain = sum(max(0.0, HEALTHY_MEAN - baseline_scores[q]) for q in v) / total_q
        out_lines.append(f"| {k} | {len(v)} | {sum_base:.4f} | +{gain:.4f} | {mean_baseline + gain:.4f} |")
        
    # Oracle Combinations
    # Combination 1: Prompt optimization (Category D only)
    gain_d = sum(max(0.0, HEALTHY_MEAN - baseline_scores[q]) for q in categories["D"]) / total_q
    out_lines.append(f"\n- **Combined Intervention 1 (Prompt only - Category D)**: Ceiling = `+{gain_d:.4f}` METEOR")
    
    # Combination 2: Context expansion + Prompt optimization (Categories C + E + D)
    gain_ced = sum(max(0.0, HEALTHY_MEAN - baseline_scores[q]) for q in categories["C"] + categories["E"] + categories["D"]) / total_q
    out_lines.append(f"- **Combined Intervention 2 (Context Expansion + Prompt - C + E + D)**: Ceiling = `+{gain_ced:.4f}` METEOR")
    
    # Combination 3: Retrieval + Context + Prompt (All categories A + B + C + D + E)
    gain_all = sum(max(0.0, HEALTHY_MEAN - baseline_scores[q]) for q in categories["A"] + categories["B"] + categories["C"] + categories["D"] + categories["E"]) / total_q
    out_lines.append(f"- **Combined Intervention 3 (Perfect Oracle - A + B + C + D + E)**: Ceiling = `+{gain_all:.4f}` METEOR")
    
    out_lines.append("")
    out_lines.append("## 4. Verdict")
    if gain_ced > 0.028:
        out_lines.append("**GO** (Theoretical ceiling of combined Context + Prompt is above +0.028)")
    else:
        out_lines.append("**NO-GO**")
        out_lines.append("Even combining all context builder and generation prompt interventions (Categories C + E + D) yields a theoretical ceiling below the 1 SE threshold of +0.028 METEOR. Therefore, we declare a final STOP.")
        
    Path(ROOT / "data/evaluation/final_causal_audit.md").write_text("\n".join(out_lines), encoding="utf-8")
    print("Report written to data/evaluation/final_causal_audit.md")
    
    conn.close()

if __name__ == "__main__":
    main()
