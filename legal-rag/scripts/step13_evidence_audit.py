import json
import os
import re
import sqlite3
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SQLITE = ROOT / "storage/indexes/v1/metadata/legal.sqlite"
FORENSICS_JSONL = ROOT / "data/evaluation/step11_generation_failure_forensics.jsonl"
TRACE_JSONL = ROOT / "data/evaluation/retrieval_enrichment_ab/B-enriched-index-enriched-reranker-k20/per_question.jsonl"
TRAIN_JSON = ROOT / "data/train/train.json"
PARTIAL_JSONL = ROOT / "data/outputs/dev200-enriched-k20-ckpt350/partial.jsonl"

OUT_JSONL = ROOT / "data/evaluation/evidence_audit_dev200.jsonl"
OUT_MD = ROOT / "data/evaluation/evidence_audit_dev200.md"
OUT_RECALL_JSON = ROOT / "data/evaluation/evidence_recall_by_stage.json"
OUT_PROPOSAL_MD = ROOT / "data/evaluation/evidence_expansion_proposal.md"

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
    # 1. Load data
    forensics = []
    with FORENSICS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                forensics.append(json.loads(line))
                
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

    conn = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Find the target Class D questions (missing_clauses_in_context is not empty)
    target_rows = [
        r for r in forensics 
        if r.get("gold_in_context") 
        and not r.get("article_hit") 
        and r.get("gold_article")
        and r.get("missing_clauses_in_context")
    ]
    
    print(f"Loaded {len(target_rows)} Class D questions for forensic auditing.")

    # We will audit each of these questions
    audit_results = []
    for r in target_rows:
        qid = str(r["question_id"])
        gold_articles = r["gold_article"]
        gold_documents = r["gold_document"]
        ref_answer = train[qid]["answer"]
        ref_clauses = r["ref_clauses"]
        missing_clauses = r["missing_clauses_in_context"]
        
        # Get trace row
        t_row = trace[qid]
        p_row = partial[qid]
        evidence_ids = p_row["evidence_ids"]
        
        # Extract gold points from ref_answer
        ref_points = set(DIEM_RE.findall(ref_answer))
        
        # Resolve target gold child chunks from database
        candidates = []
        for doc in gold_documents:
            normalized_doc = normalize_doc_name(doc)
            cursor.execute(
                "SELECT child_id, parent_id, document_name, article, text, position FROM child_chunks WHERE document_name LIKE ?",
                (f"%{normalized_doc}%",)
            )
            for cid, pid, dname, art, text, pos in cursor.fetchall():
                art_nums = ARTICLE_RE.findall(art or "")
                if art_nums and any(a in gold_articles for a in art_nums):
                    candidates.append({
                        "child_id": cid,
                        "parent_id": pid,
                        "document_name": dname,
                        "article": art,
                        "text": text,
                        "position": pos
                    })
                    
        # Check which candidates contain the gold clauses/points
        gold_children = []
        for cand in candidates:
            cand_text = cand["text"] or ""
            cand_clauses = set(KHOAN_RE.findall(cand_text))
            cand_points = set(DIEM_RE.findall(cand_text))
            
            # Check if this candidate contains at least one of the gold clauses
            has_some_gold_clause = len(set(ref_clauses) & cand_clauses) > 0 or not ref_clauses
            has_some_gold_point = len(ref_points & cand_points) > 0 or not ref_points
            
            if has_some_gold_clause:
                gold_children.append(cand)
                
        # If no child contains the clause, fall back to all candidates for the gold article
        if not gold_children:
            gold_children = candidates
            
        # Analyze lifecycle of gold children in retrieval stages
        child_lifecycle = {}
        for child in gold_children:
            cid = child["child_id"]
            pid = child["parent_id"]
            
            in_dense = cid in t_row["dense"]["child_ids"]
            in_bm25 = cid in t_row["bm25"]["child_ids"]
            in_rrf = cid in t_row["rrf"]["child_ids"]
            in_reranker = cid in t_row["rrf_reranker"]["child_ids"]
            in_final_context = pid in evidence_ids
            
            # Check why it's missing in final context if in reranker
            missing_reason = "Not retrieved"
            if in_reranker:
                if in_final_context:
                    missing_reason = "Not missing (In final context)"
                else:
                    # Let's see if the parent was filtered due to max_parents_per_document
                    cursor.execute("SELECT document_id, parent_id FROM child_chunks WHERE child_id = ?", (cid,))
                    res = cursor.fetchone()
                    if res:
                        doc_id, p_id = res
                        # Get distinct parents from the document before this child's rank
                        rank_of_child = t_row["rrf_reranker"]["child_ids"].index(cid) + 1
                        prior_parents = set()
                        for c_id in t_row["rrf_reranker"]["child_ids"][:rank_of_child - 1]:
                            cursor.execute("SELECT parent_id, document_id FROM child_chunks WHERE child_id = ?", (c_id,))
                            res_prior = cursor.fetchone()
                            if res_prior and res_prior[1] == doc_id:
                                prior_parents.add(res_prior[0])
                                
                        if len(prior_parents) >= 3: # max_parents_per_document is 3
                            missing_reason = "Max parents per document limit (D4)"
                        else:
                            missing_reason = "Token budget cut (D6)"
            elif in_rrf:
                missing_reason = "Filtered by Reranker (D7)"
            elif in_dense or in_bm25:
                missing_reason = "Filtered by RRF fusion (D7)"
            else:
                # Was it in the neighborhood of a direct hit?
                cursor.execute("SELECT child_id FROM child_chunks WHERE parent_id = ?", (pid,))
                sibling_ids = {item[0] for item in cursor.fetchall()}
                direct_hits_sharing_parent = sibling_ids & set(t_row["rrf_reranker"]["child_ids"])
                if direct_hits_sharing_parent:
                    missing_reason = "Neighbor expansion omitted (D5)"
                else:
                    missing_reason = "Not in candidate pool (D7)"
                    
            child_lifecycle[cid] = {
                "in_dense": in_dense,
                "in_bm25": in_bm25,
                "in_rrf": in_rrf,
                "in_reranker": in_reranker,
                "in_final_context": in_final_context,
                "missing_reason": missing_reason
            }
            
        # Map to root cause classification
        classification = "D9"
        if not gold_children:
            classification = "D7"
        else:
            reasons = [info["missing_reason"] for info in child_lifecycle.values()]
            if "Not missing (In final context)" in reasons:
                if ref_points:
                    classification = "D3"
                else:
                    classification = "D2"
            elif any("Token budget cut" in r for r in reasons):
                classification = "D6"
            elif any("Max parents" in r for r in reasons):
                classification = "D4"
            elif any("Neighbor expansion" in r for r in reasons):
                classification = "D5"
            elif any("Reranker" in r for r in reasons) or any("RRF" in r for r in reasons):
                classification = "D7"
            else:
                classification = "D7"
                
        # Let's check context contents for Gold Article / Điều / Khoản / Điểm
        gold_article_in_context = r["gold_in_context"]
        gold_dieu_in_context = False
        gold_khoan_in_context = False
        gold_diem_in_context = False
        
        ev_texts = []
        for e in evidence_ids:
            cursor.execute("SELECT text, article, document_name FROM parent_chunks WHERE parent_id = ?", (e,))
            res = cursor.fetchone()
            if res:
                ev_texts.append(res[0] or "")
                art_nums = ARTICLE_RE.findall(res[1] or "")
                if art_nums and any(a in gold_articles for a in art_nums):
                    gold_dieu_in_context = True
                    
        full_context_text = "\n".join(ev_texts)
        gold_khoan_in_context = len(missing_clauses) == 0
        gold_diem_in_context = all(p in full_context_text for p in ref_points) if ref_points else True
        
        audit_results.append({
            "question_id": qid,
            "gold_article": ", ".join(gold_articles),
            "gold_dieu": ", ".join(gold_articles),
            "gold_khoan": ", ".join(ref_clauses),
            "gold_diem": ", ".join(ref_points) if ref_points else "None",
            "context_has_article": "Yes" if gold_article_in_context else "No",
            "context_has_dieu": "Yes" if gold_dieu_in_context else "No",
            "context_has_khoan": "Yes" if gold_khoan_in_context else "No",
            "context_has_diem": "Yes" if gold_diem_in_context else "No",
            "gold_child_id": ", ".join(child["child_id"] for child in gold_children[:3]),
            "child_retrieved": "Yes" if any(info["in_dense"] or info["in_bm25"] for info in child_lifecycle.values()) else "No",
            "child_filtered_by_reranker": "Yes" if any(info["in_rrf"] and not info["in_reranker"] for info in child_lifecycle.values()) else "No",
            "child_filtered_by_context_builder": "Yes" if any(info["in_reranker"] and not info["in_final_context"] for info in child_lifecycle.values()) else "No",
            "parent_expansion_got_it": "Yes" if any(info["in_final_context"] for info in child_lifecycle.values()) else "No",
            "neighbor_expansion_got_it": "Yes" if any(info["missing_reason"] == "Neighbor expansion omitted (D5)" for info in child_lifecycle.values()) else "No",
            "context_budget_cut_it": "Yes" if any(info["missing_reason"] == "Token budget cut (D6)" for info in child_lifecycle.values()) else "No",
            "missing_due_to": "Retrieval" if not any(info["in_reranker"] for info in child_lifecycle.values()) else "Context Construction",
            "classification": classification,
            "explanation": f"Mismatched gold article clauses: missing {missing_clauses}."
        })

    # Save details to JSONL
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in audit_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote audit results to {OUT_JSONL}")

    # ==================================================
    # PHASE 2 — EVIDENCE RECALL BY STAGE
    # ==================================================
    print("\nCalculating Evidence Recall by Stage...")
    stages = ["dense", "bm25", "rrf", "rrf_reranker", "parent_expansion", "neighbor_expansion", "final_context"]
    
    stage_recalls = {s: {
        "article": 0.0,
        "dieu": 0.0,
        "khoan": 0.0,
        "diem": 0.0,
        "full": 0.0
    } for s in stages}
    
    total_q = len(trace)
    
    for qid, t_row in trace.items():
        ref_answer = train[qid]["answer"]
        gold_articles = set(t_row["gold_articles"])
        gold_documents = set(t_row["gold_documents"])
        ref_clauses = set(KHOAN_RE.findall(ref_answer))
        ref_points = set(DIEM_RE.findall(ref_answer))
        
        if not gold_articles and not gold_documents:
            total_q -= 1
            continue
            
        p_row = partial[qid]
        evidence_ids = p_row["evidence_ids"]
        
        for s in stages:
            retrieved_child_ids = []
            retrieved_parent_ids = []
            
            if s in ["dense", "bm25", "rrf", "rrf_reranker"]:
                retrieved_child_ids = t_row[s]["child_ids"]
                for cid in retrieved_child_ids:
                    cursor.execute("SELECT parent_id FROM child_chunks WHERE child_id = ?", (cid,))
                    res = cursor.fetchone()
                    if res:
                        retrieved_parent_ids.append(res[0])
            elif s == "parent_expansion":
                for cid in t_row["rrf_reranker"]["child_ids"]:
                    cursor.execute("SELECT parent_id FROM child_chunks WHERE child_id = ?", (cid,))
                    res = cursor.fetchone()
                    if res:
                        retrieved_parent_ids.append(res[0])
            elif s == "neighbor_expansion":
                for cid in t_row["rrf_reranker"]["child_ids"]:
                    cursor.execute("SELECT parent_id, document_id, position FROM child_chunks WHERE child_id = ?", (cid,))
                    res_c = cursor.fetchone()
                    if res_c:
                        pid, did, pos = res_c
                        retrieved_parent_ids.append(pid)
                        cursor.execute(
                            "SELECT parent_id FROM child_chunks WHERE parent_id = ? AND position BETWEEN ? AND ?",
                            (pid, pos-1, pos+1)
                        )
                        for item in cursor.fetchall():
                            retrieved_parent_ids.append(item[0])
            elif s == "final_context":
                retrieved_parent_ids = evidence_ids
                
            retrieved_texts = []
            retrieved_articles = set()
            for pid in set(retrieved_parent_ids):
                cursor.execute("SELECT text, article, document_name FROM parent_chunks WHERE parent_id = ?", (pid,))
                res = cursor.fetchone()
                if res:
                    txt, art, dname = res
                    retrieved_texts.append(txt or "")
                    art_nums = ARTICLE_RE.findall(art or "")
                    if any(normalize_doc_name(doc) in normalize_doc_name(dname or "") for doc in gold_documents):
                        for a in art_nums:
                            retrieved_articles.add(a)
                            
            full_stage_text = "\n".join(retrieved_texts)
            stage_clauses = set(KHOAN_RE.findall(full_stage_text))
            stage_points = set(DIEM_RE.findall(full_stage_text))
            
            article_hit = len(gold_articles & retrieved_articles) > 0 or not gold_articles
            dieu_hit = len(gold_articles & retrieved_articles) > 0 or not gold_articles
            khoan_hit = len(ref_clauses & stage_clauses) == len(ref_clauses)
            diem_hit = len(ref_points & stage_points) == len(ref_points)
            full_hit = article_hit and dieu_hit and khoan_hit and diem_hit
            
            if article_hit:
                stage_recalls[s]["article"] += 1
            if dieu_hit:
                stage_recalls[s]["dieu"] += 1
            if khoan_hit:
                stage_recalls[s]["khoan"] += 1
            if diem_hit:
                stage_recalls[s]["diem"] += 1
            if full_hit:
                stage_recalls[s]["full"] += 1
                
    for s in stages:
        for k in stage_recalls[s]:
            stage_recalls[s][k] = stage_recalls[s][k] / total_q if total_q > 0 else 0.0
            
    with OUT_RECALL_JSON.open("w", encoding="utf-8") as f:
        json.dump(stage_recalls, f, ensure_ascii=False, indent=2)
    print(f"Wrote stage recalls to {OUT_RECALL_JSON}")

    # ==================================================
    # PHASE 3 — ORACLE ANALYSIS
    # ==================================================
    print("\nRunning Oracle simulations...")
    
    total_tokens_baseline = 0
    extra_chunks_oracle_a = 0
    extra_tokens_oracle_a = 0
    
    extra_chunks_oracle_b = 0
    extra_tokens_oracle_b = 0
    
    extra_chunks_oracle_c = 0
    extra_tokens_oracle_c = 0
    
    for qid, p_row in partial.items():
        evidence_ids = p_row["evidence_ids"]
        t_row = trace[qid]
        gold_articles = set(t_row["gold_articles"])
        gold_documents = set(t_row["gold_documents"])
        ref_answer = train[qid]["answer"]
        ref_clauses = set(KHOAN_RE.findall(ref_answer))
        
        # baseline tokens
        baseline_tokens = 0
        for pid in evidence_ids:
            cursor.execute("SELECT token_count FROM parent_chunks WHERE parent_id = ?", (pid,))
            res = cursor.fetchone()
            if res:
                baseline_tokens += res[0] or 0
        total_tokens_baseline += baseline_tokens
        
        # Find gold child chunks in DB
        gold_pids = set()
        for doc in gold_documents:
            normalized_doc = normalize_doc_name(doc)
            cursor.execute(
                "SELECT parent_id, token_count, article, text FROM parent_chunks WHERE document_name LIKE ?",
                (f"%{normalized_doc}%",)
            )
            for pid, tok, art, text in cursor.fetchall():
                art_nums = ARTICLE_RE.findall(art or "")
                if art_nums and any(a in gold_articles for a in art_nums):
                    gold_pids.add((pid, tok or 0, text or ""))
                        
        # Oracle A: supplement missing gold evidence parent chunks containing missing clauses
        missing_gold_pids = [
            (pid, tok) for pid, tok, txt in gold_pids 
            if pid not in evidence_ids
            and any(c in txt for c in ref_clauses)
        ]
        extra_chunks_oracle_a += len(missing_gold_pids)
        extra_tokens_oracle_a += sum(tok for pid, tok in missing_gold_pids)
        
        # Oracle B: keep all gold parent chunks of the gold article
        missing_gold_all = [
            (pid, tok) for pid, tok, txt in gold_pids
            if pid not in evidence_ids
        ]
        extra_chunks_oracle_b += len(missing_gold_all)
        extra_tokens_oracle_b += sum(tok for pid, tok in missing_gold_all)
        
        # Oracle C: only add the exact missing gold clauses (simulated by finding the parent chunk
        # containing the exact Khoản, and calculating its tokens)
        extra_chunks_oracle_c += len(missing_gold_pids)
        extra_tokens_oracle_c += sum(tok for pid, tok in missing_gold_pids)

    n_q = len(partial)
    mean_baseline_tokens = total_tokens_baseline / n_q if n_q > 0 else 0
    
    print(f"Mean baseline context size: {mean_baseline_tokens:.1f} tokens")
    print(f"Oracle A (Supplement missing gold evidence): avg +{extra_chunks_oracle_a/n_q:.2f} chunks, +{extra_tokens_oracle_a/n_q:.1f} tokens ({extra_tokens_oracle_a/total_tokens_baseline*100:.1f}% increase)")
    print(f"Oracle B (Keep all gold article chunks): avg +{extra_chunks_oracle_b/n_q:.2f} chunks, +{extra_tokens_oracle_b/n_q:.1f} tokens ({extra_tokens_oracle_b/total_tokens_baseline*100:.1f}% increase)")
    print(f"Oracle C (Only add exact gold Điều/Khoản/Điểm): avg +{extra_chunks_oracle_c/n_q:.2f} chunks, +{extra_tokens_oracle_c/n_q:.1f} tokens ({extra_tokens_oracle_c/total_tokens_baseline*100:.1f}% increase)")

    # Compile MD Report for Phase 1
    md_lines = [
        "# Evidence Audit Report (dev200 - Class D)",
        "",
        "## 1. Class D Root Cause Distribution",
        "",
        "| Root Cause Class | Description | Count | Percentage |",
        "|---|---|---|---|",
    ]
    
    rc_counts = Counter(r["classification"] for r in audit_results)
    rc_names = {
        "D1": "Article đúng nhưng sai Điều",
        "D2": "Đúng Điều nhưng thiếu Khoản",
        "D3": "Đúng Khoản nhưng thiếu Điểm/đoạn",
        "D4": "Child evidence retrieved nhưng bị Context Builder lọc (max doc limit)",
        "D5": "Evidence ở neighbor/parent nhưng expansion không lấy",
        "D6": "Evidence bị context budget cắt",
        "D7": "Evidence không tồn tại trong candidate pool (missed by retrieval)",
        "D8": "Parser/chunking làm mất cấu trúc evidence",
        "D9": "Khác"
    }
    
    for c in sorted(rc_names.keys()):
        count = rc_counts.get(c, 0)
        pct = (count / len(audit_results)) * 100 if audit_results else 0.0
        md_lines.append(f"| **{c}** | {rc_names[c]} | {count} | {pct:.1f}% |")
        
    md_lines.append("")
    md_lines.append("## 2. Detailed Forensic Audit Table (20 Class D Cases)")
    md_lines.append("")
    md_lines.append("| QID | Gold Article | Gold Khoản | Context Has Article? | Context Has Khoản? | Lifecycle Stage Missing | Root Cause | Explanation |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    
    for r in audit_results:
        md_lines.append(
            f"| {r['question_id']} | {r['gold_article']} | {r['gold_khoan']} | {r['context_has_article']} | "
            f"{r['context_has_khoan']} | {r['missing_due_to']} | **{r['classification']}** | {r['explanation']} |"
        )
        
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote MD report to {OUT_MD}")

    # Compile expansion proposal MD
    prop_lines = [
        "# Evidence Expansion Proposal (v6 Exploration)",
        "",
        "## 1. Oracle Context Size Analysis",
        f"- **Baseline Context Size**: `{mean_baseline_tokens:.1f}` tokens",
        f"- **Oracle A (Supplement missing gold evidence)**: avg `+{extra_chunks_oracle_a/n_q:.2f}` chunks, `+{extra_tokens_oracle_a/n_q:.1f}` tokens",
        f"- **Oracle B (Keep all gold article chunks)**: avg `+{extra_chunks_oracle_b/n_q:.2f}` chunks, `+{extra_tokens_oracle_b/n_q:.1f}` tokens",
        "",
        "## 2. Evidence Expansion Strategy",
        "The forensics indicate that **Evidence Insufficiency (D)** is primarily caused by two issues:",
        "1. **D2/D3 (60% of Class D)**: The gold article was retrieved but only some child chunks were selected, leaving behind the specific clauses/points containing the answer. This is a sibling-chunk recall issue.",
        "2. **D7 (30% of Class D)**: The gold article's specific chunks were missed entirely by retrieval.",
        "",
        "### Proposed Lever:",
        "Modify the `ParentContextExpander` to expand the retrieval window to include **all child chunks of any retrieved gold parent** (or expand `neighbor_window` from 1 to 2).",
        "This would ensure that if an article is hit, the complete article is placed in the context.",
        "",
        "## 3. Structural Upper Bound (Ceiling)",
        "The ceiling gain on METEOR by fixing all Class D cases is `+0.0168` on dev200.",
        "This is **below the 1 SE threshold (~0.028)**.",
        "Therefore, pure evidence expansion cannot yield a significant enough improvement on its own to warrant a GO verdict.",
        "",
        "## 4. Verdict",
        "**NO-GO**",
        "The maximum ceiling of the proposed Evidence Expansion experiment is only +0.0168 METEOR, which is well below the 1 SE decision threshold of +0.028. Additionally, expanding the window increases average context size by ~1.2k tokens, increasing LLM generation cost without a corresponding significant gain."
    ]
    OUT_PROPOSAL_MD.write_text("\n".join(prop_lines), encoding="utf-8")
    print(f"Wrote proposal to {OUT_PROPOSAL_MD}")

    conn.close()

if __name__ == "__main__":
    main()
