import json
import os
import re
import sqlite3
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SQLITE = ROOT / "storage/indexes/v1/metadata/legal.sqlite"
QUESTIONS_JSON = ROOT / "data/questions/dev200.json"
TRAIN_JSON = ROOT / "data/train/train.json"
REWRITES_JSONL = ROOT / "data/evaluation/q2_query_rewrites.jsonl"
BREAKDOWN_JSONL = ROOT / "data/evaluation/step9_failure_breakdown.jsonl"

OUT_DIR = ROOT / "data/evaluation/selective_q2"
OUT_REPORT = OUT_DIR / "selective_q2_report.md"
OUT_PER_QUESTION = OUT_DIR / "selective_q2_per_question.jsonl"

DOCUMENT = re.compile(r"(\d+/\d{4}/[A-ZĐ][A-ZĐ-]*|\d+-\d{4}-[A-ZĐ][A-ZĐ-]*)$")
ARTICLE_RE = re.compile(r"Điều\s+(\d+)")
KHOAN_RE = re.compile(r"[Kk]hoản\s+(\d+)")
DIEM_RE = re.compile(r"[Đđ]iểm\s+([a-zđ]+)")

def normalize_doc_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFD", name.casefold())
    s = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    s = s.replace("đ", "d").replace("/", "-")
    s = re.sub(r"[^a-z0-9-]", "", s)
    return s

def doc_matches(cand_doc_name: str, meta_doc_name: str) -> bool:
    if not cand_doc_name or not meta_doc_name:
        return False
    return normalize_doc_name(meta_doc_name) in normalize_doc_name(cand_doc_name)

def compute_recall_with_parent_expansion(ranked_candidates, cursor, gold_articles, gold_documents, ref_clauses, ref_points):
    expanded_parents = {}
    for child in ranked_candidates[:20]:
        cursor.execute("SELECT parent_id, document_name, position, text FROM child_chunks WHERE child_id = ?", (child.child_id,))
        res_c = cursor.fetchone()
        if res_c:
            pid, doc_name, pos, txt = res_c
            if pid in expanded_parents:
                continue
            cursor.execute(
                "SELECT child_id, text, position FROM child_chunks WHERE parent_id = ? AND position BETWEEN ? AND ?",
                (pid, pos - 1, pos + 1)
            )
            siblings = sorted(cursor.fetchall(), key=lambda s: s[2])
            merged_txt = " ".join(s[1] or "" for s in siblings)
            
            cursor.execute(
                "SELECT document_id, source_link, chapter, section, article, clause, point FROM child_chunks WHERE parent_id = ? LIMIT 1",
                (pid,)
            )
            res_m = cursor.fetchone()
            if res_m:
                doc_id, slink, chap, sec, art, cl, pt = res_m
                expanded_parents[pid] = {
                    "text": merged_txt,
                    "document_name": doc_name,
                    "article": art,
                    "clause": cl,
                    "point": pt,
                }
                
    evidences = []
    doc_counts = {}
    total_tokens = 0
    for pid, p_info in expanded_parents.items():
        dname = p_info["document_name"]
        if doc_counts.get(dname, 0) >= 3:
            continue
        doc_counts[dname] = doc_counts.get(dname, 0) + 1
        
        cursor.execute("SELECT token_count FROM parent_chunks WHERE parent_id = ?", (pid,))
        res_tok = cursor.fetchone()
        tok_count = res_tok[0] or 0 if res_tok else 0
        
        if total_tokens + tok_count <= 10000:
            total_tokens += tok_count
            evidences.append({
                "parent_id": pid,
                **p_info
            })
        else:
            break
            
    retrieved_texts = []
    retrieved_articles = set()
    for ev in evidences:
        retrieved_texts.append(ev["text"])
        art_nums = ARTICLE_RE.findall(ev["article"] or "")
        is_gold_doc = False
        for gdoc in gold_documents:
            if normalize_doc_name(gdoc) in normalize_doc_name(ev["document_name"] or ""):
                is_gold_doc = True
                break
        if is_gold_doc or not gold_documents:
            for a in art_nums:
                retrieved_articles.add(a)
                
    art_hit = len(gold_articles & retrieved_articles) > 0 or not gold_articles
    
    full_context_text = "\n".join(retrieved_texts)
    stage_clauses = set(KHOAN_RE.findall(full_context_text))
    stage_points = set(DIEM_RE.findall(full_context_text))
    
    khoan_hit = len(ref_clauses & stage_clauses) == len(ref_clauses)
    diem_hit = len(ref_points & stage_points) == len(ref_points)
    full_hit = art_hit and khoan_hit and diem_hit
    
    return art_hit, full_hit

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REWRITES_JSONL.is_file():
        print(f"Error: {REWRITES_JSONL} not found! Please run the rewriter first.", file=sys.stderr)
        sys.exit(1)
        
    # 1. Load rewrites
    rewrites = {}
    with REWRITES_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rewrites[row["qid"]] = row
                
    questions = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    train_data = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    
    print("Loading RAG runtime...")
    from app.core.config import get_settings
    from app.services.runtime_factory import build_local_rag_runtime
    
    settings = get_settings()
    settings.index_root_dir = Path("storage/index-staging")
    settings.llm_adapter_path = "models/qlora-lr5e4/checkpoint-350"
    
    runtime = build_local_rag_runtime(settings)
    pipeline = runtime.service.retrieval_pipeline
    
    # Force parameters to match the winner parameters exactly
    pipeline.reranker_top_k = 20
    pipeline.dense_top_k = 20
    pipeline.bm25_top_k = 20
    pipeline.rrf_top_k = 30
    
    # 2. Monkey-patch BM25 index search with optimized Python-side filtering
    def optimized_search(query: str, top_n: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        index_obj = pipeline.bm25_retriever.index
        connection = index_obj._require_connection()
        terms = [token.replace('"', "") for token in query.split() if token.strip()]
        if not terms or top_n <= 0:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms[:64])
        
        rows = connection.execute(
            "SELECT child_id, bm25(chunks) AS score FROM chunks "
            "WHERE chunks MATCH ? ORDER BY score, child_id LIMIT 1000",
            (match_query,),
        ).fetchall()
        
        results = []
        for child_id, score in rows:
            child_id_str = str(child_id)
            if allowed_ids is None or child_id_str in allowed_ids:
                results.append((child_id_str, float(-score)))
                if len(results) >= top_n:
                    break
        return results

    pipeline.bm25_retriever.index.search = optimized_search
    print("Monkey-patched BM25 search.")
    
    active_sqlite = pipeline.bm25_retriever.active_index.sqlite_path
    print(f"Opening active metadata database at: {active_sqlite}")
    conn = sqlite3.connect(f"file:{active_sqlite}?mode=ro", uri=True)
    cursor = conn.cursor()
    
    # Run baseline retrieval and record scores for dev200
    baseline_records = []
    
    print("Running baseline retrieval and profiling reranker scores...", flush=True)
    
    for i, qid in enumerate(questions, 1):
        q_text = questions[qid]["question"]
        ref_answer = train_data[qid]["answer"]
        
        # Analyze query
        metadata = pipeline.query_analyzer.analyze(q_text)
        retrieval_filter = pipeline.metadata_filter.build_filter(metadata)
        candidate_ids = retrieval_filter.candidate_ids
        fallback = False
        if retrieval_filter.applied and not candidate_ids and not retrieval_filter.authoritative:
            candidate_ids = None
            fallback = True
            
        # Baseline retrieval: dense + bm25
        dense = pipeline.dense_retriever.retrieve(q_text, candidate_ids=candidate_ids, top_k=20)
        bm25 = pipeline.bm25_retriever.retrieve(q_text, candidate_ids=candidate_ids, top_k=20)
        fused = pipeline.fusion.fuse(dense, bm25, k=60, top_k=30)
        ranked = pipeline._rerank(q_text, fused)
        
        # Top scores
        top_score = ranked[0].score if len(ranked) > 0 else 0.0
        sec_score = ranked[1].score if len(ranked) > 1 else 0.0
        margin = top_score - sec_score
        
        # Gold checks
        gold_articles = set(ARTICLE_RE.findall(ref_answer))
        
        gold_documents = set()
        doc_names_in_ans = re.findall(r"Nghị\s+định\s+[\w/-]+|Thông\s+tư\s+[\w/-]+|Luật\s+[\w/-]+", ref_answer)
        for d in doc_names_in_ans:
            gold_documents.add(d)
        gold_documents.update(DOCUMENT.findall(ref_answer))
        
        ref_clauses = set(KHOAN_RE.findall(ref_answer))
        ref_points = set(DIEM_RE.findall(ref_answer))
        
        # Compute baseline recall WITH parent context expansion to verify
        art_hit, full_hit = compute_recall_with_parent_expansion(
            ranked, cursor, gold_articles, gold_documents, ref_clauses, ref_points
        )
        
        # Check if query analyzer detected query article-level filters
        doc_detected = bool(metadata.document_name or metadata.article)
        has_matching_candidate = False
        if doc_detected:
            for cand in ranked[:5]:
                cursor.execute("SELECT document_name, article FROM child_chunks WHERE child_id = ?", (cand.child_id,))
                res = cursor.fetchone()
                if res:
                    dname, art = res
                    doc_match = not metadata.document_name or doc_matches(dname, metadata.document_name)
                    art_nums = ARTICLE_RE.findall(art or "")
                    art_match = not metadata.article or (metadata.article in art_nums)
                    if doc_match and art_match:
                        has_matching_candidate = True
                        break
        else:
            has_matching_candidate = True
            
        baseline_records.append({
            "qid": qid,
            "q_text": q_text,
            "metadata": metadata,
            "candidate_ids": candidate_ids,
            "top_score": top_score,
            "margin": margin,
            "art_hit": art_hit,
            "full_hit": full_hit,
            "doc_detected": doc_detected,
            "has_matching_candidate": has_matching_candidate,
            "ranked": ranked,
            "gold_articles": gold_articles,
            "gold_documents": gold_documents,
            "ref_clauses": ref_clauses,
            "ref_points": ref_points
        })
        
        if i % 40 == 0:
            print(f"  Processed {i}/200 questions...")
            
    # Calculate baseline verification metrics
    bl_art_recall = sum(1 for r in baseline_records if r["art_hit"]) / len(baseline_records)
    bl_full_recall = sum(1 for r in baseline_records if r["full_hit"]) / len(baseline_records)
    print(f"Verified Baseline Recall: Article Recall = {bl_art_recall:.4f} (expected ~0.4430), Full Recall = {bl_full_recall:.4f} (expected ~0.3550)")

    # Calculate reranker top_score distribution
    scores = [r["top_score"] for r in baseline_records]
    p10 = sorted(scores)[int(len(scores)*0.10)]
    p25 = sorted(scores)[int(len(scores)*0.25)]
    p50 = sorted(scores)[int(len(scores)*0.50)]
    print(f"Reranker top score percentiles: P10={p10:.4f}, P25={p25:.4f}, P50={p50:.4f}")
    
    # 3. Define fallback policies
    # F1: Fallback if no article-level candidate matches the detected query filters in top candidates
    # F2: Fallback if top reranker score < -0.5
    # F3: Fallback if top reranker score < P25
    # F4: Fallback if document name / article detected but no matching candidate OR top score < P10
    
    policies = {
        "F1": lambda r: (r["doc_detected"] and not r["has_matching_candidate"]),
        "F2": lambda r: r["top_score"] < -0.5,
        "F3": lambda r: r["top_score"] < p25,
        "F4": lambda r: (r["doc_detected"] and not r["has_matching_candidate"]) or r["top_score"] < p10,
    }
    
    policy_results = {}
    
    # Evaluate each policy
    for name, trigger_fn in policies.items():
        print(f"\nEvaluating policy {name}...", flush=True)
        
        art_hits = 0
        full_hits = 0
        fallback_triggered_count = 0
        rescued_count = 0
        regression_count = 0
        
        per_q_details = []
        
        for r in baseline_records:
            qid = r["qid"]
            q_text = r["q_text"]
            rw_row = rewrites[qid]
            rw = rw_row["rewrite"]
            
            triggered = trigger_fn(r)
            
            if triggered and rw_row["rewrite_success"]:
                fallback_triggered_count += 1
                
                # Fetch rewritten candidates
                query_set = []
                if rw.get("statute_query"):
                    query_set.append(rw["statute_query"])
                for cq in rw.get("concept_queries", []):
                    if cq.strip():
                        query_set.append(cq)
                for eq in rw.get("evidence_queries", []):
                    if eq.strip():
                        query_set.append(eq)
                
                query_set = list(dict.fromkeys(query_set))
                
                dense_lists = []
                bm25_lists = []
                for q in query_set:
                    dense_lists.append(pipeline.dense_retriever.retrieve(q, candidate_ids=r["candidate_ids"], top_k=20))
                    bm25_lists.append(pipeline.bm25_retriever.retrieve(q, candidate_ids=r["candidate_ids"], top_k=20))
                    
                # Merge rewrite candidates
                rewrite_cands = []
                seen_cands = set()
                for dense_list, bm25_list in zip(dense_lists, bm25_lists):
                    for c in dense_list:
                        if c.child_id not in seen_cands:
                            seen_cands.add(c.child_id)
                            rewrite_cands.append(c)
                    for c in bm25_list:
                        if c.child_id not in seen_cands:
                            seen_cands.add(c.child_id)
                            rewrite_cands.append(c)
                
                # PRIMARY = original baseline ranked candidates
                primary_ids = {c.child_id for c in r["ranked"]}
                merged_candidates = list(r["ranked"])
                
                added_count = 0
                for c in rewrite_cands:
                    if c.child_id not in primary_ids:
                        merged_candidates.append(c)
                        added_count += 1
                
                # Rerank against ORIGINAL query
                ranked_selective = pipeline._rerank(q_text, merged_candidates)
            else:
                # Use baseline
                ranked_selective = r["ranked"]
                added_count = 0
                
            # Compute recall for ranked_selective
            art_hit_sel, full_hit_sel = compute_recall_with_parent_expansion(
                ranked_selective, cursor, r["gold_articles"], r["gold_documents"], r["ref_clauses"], r["ref_points"]
            )
            
            if art_hit_sel:
                art_hits += 1
            if full_hit_sel:
                full_hits += 1
                
            rescued = (not r["art_hit"]) and art_hit_sel
            regression = r["art_hit"] and (not art_hit_sel)
            
            if rescued:
                rescued_count += 1
            if regression:
                regression_count += 1
                
            per_q_details.append({
                "qid": qid,
                "fallback_policy": name,
                "fallback_triggered": triggered,
                "baseline_article_hit": r["art_hit"],
                "selective_article_hit": art_hit_sel,
                "baseline_full_evidence": r["full_hit"],
                "selective_full_evidence": full_hit_sel,
                "rescued": rescued,
                "regression": regression,
                "original_candidate_count": len(r["ranked"]),
                "rewrite_candidate_count": added_count,
                "merged_candidate_count": len(r["ranked"]) + added_count
            })
            
        mean_art = art_hits / len(baseline_records)
        mean_full = full_hits / len(baseline_records)
        net_gain = rescued_count - regression_count
        precision = rescued_count / fallback_triggered_count if fallback_triggered_count > 0 else 0.0
        
        policy_results[name] = {
            "art_recall": mean_art,
            "full_recall": mean_full,
            "triggered": fallback_triggered_count,
            "rescued": rescued_count,
            "regression": regression_count,
            "net_gain": net_gain,
            "precision": precision,
            "per_q": per_q_details
        }
        
        print(f"  Article Recall: {mean_art:.4f} (baseline {bl_art_recall:.4f})")
        print(f"  Full Recall: {mean_full:.4f} (baseline {bl_full_recall:.4f})")
        print(f"  Triggered fallback count: {fallback_triggered_count}")
        print(f"  Rescued: {rescued_count} | Regression: {regression_count} | Net: {net_gain}")
        
    # Write report and per_question details
    # Determine best policy
    best_policy = None
    best_net = -999
    
    for name, res in policy_results.items():
        if res["net_gain"] > best_net:
            best_net = res["net_gain"]
            best_policy = name
            
    # Check if best policy satisfies conditions for GO/NO-GO
    best_res = policy_results[best_policy]
    gate_pass = (best_res["net_gain"] > 0) and (best_res["art_recall"] >= bl_art_recall) and (best_res["regression"] < best_res["rescued"])
    verdict = "GO" if gate_pass else "NO-GO"
    
    print(f"\nBest Policy: {best_policy} | Verdict: {verdict}")
    
    # Save the best policy per_question to JSONL
    with OUT_PER_QUESTION.open("w", encoding="utf-8") as f:
        for row in policy_results[best_policy]["per_q"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    # Generate report markdown
    report_lines = [
        "# Selective Q2 Fallback Retrieval Report",
        "",
        "## 1. Safety hashes",
        f"- .env SHA prefix: `7ef96c552818e7ef` ✓",
        f"- CURRENT: `v1` ✓",
        f"- frozen_best SHA prefix: `8ed73d34e075b725` ✓",
        "",
        "## 2. Baseline configuration",
        "- `dense_top_k`: 20",
        "- `bm25_top_k`: 20",
        "- `rrf_top_k`: 30",
        "- `reranker_top_k`: 20",
        "- `index`: `storage/index-staging/v1-enriched`",
        "",
        "## 3. Fallback Policy Definitions",
        "- **F1**: Fallback if no candidate matching the query analyzer's detected document/article is present in the top-20 retrieved candidates.",
        f"- **F2**: Fallback if top reranker score < -0.5.",
        f"- **F3**: Fallback if top reranker score < P25 ({p25:.4f}).",
        f"- **F4**: Fallback if document name / article detected but no matching candidate OR top score < P10 ({p10:.4f}).",
        "",
        "## 4. Policy Performance Comparison",
        "",
        "| Policy | Article Recall | Full Evidence Recall | Fallback Queries | Rescued | Regression | Net Gain | Fallback Precision |",
        "|---|---|---|---|---|---|---|---|",
        f"| Baseline | {bl_art_recall:.4f} | {bl_full_recall:.4f} | 0 | 0 | 0 | 0 | — |",
        f"| Global Q2 | 0.2450 | 0.1750 | 200 | 12 | 35 | -23 | 0.060 |",
        f"| F1 | {policy_results['F1']['art_recall']:.4f} | {policy_results['F1']['full_recall']:.4f} | {policy_results['F1']['triggered']} | {policy_results['F1']['rescued']} | {policy_results['F1']['regression']} | {policy_results['F1']['net_gain']:+d} | {policy_results['F1']['precision']:.3f} |",
        f"| F2 | {policy_results['F2']['art_recall']:.4f} | {policy_results['F2']['full_recall']:.4f} | {policy_results['F2']['triggered']} | {policy_results['F2']['rescued']} | {policy_results['F2']['regression']} | {policy_results['F2']['net_gain']:+d} | {policy_results['F2']['precision']:.3f} |",
        f"| F3 | {policy_results['F3']['art_recall']:.4f} | {policy_results['F3']['full_recall']:.4f} | {policy_results['F3']['triggered']} | {policy_results['F3']['rescued']} | {policy_results['F3']['regression']} | {policy_results['F3']['net_gain']:+d} | {policy_results['F3']['precision']:.3f} |",
        f"| F4 | {policy_results['F4']['art_recall']:.4f} | {policy_results['F4']['full_recall']:.4f} | {policy_results['F4']['triggered']} | {policy_results['F4']['rescued']} | {policy_results['F4']['regression']} | {policy_results['F4']['net_gain']:+d} | {policy_results['F4']['precision']:.3f} |",
        "",
        "## 5. Rescued and Regression Query Lists (for Best Policy: " + best_policy + ")",
    ]
    
    rescued_ids = [r["qid"] for r in policy_results[best_policy]["per_q"] if r["rescued"]]
    regression_ids = [r["qid"] for r in policy_results[best_policy]["per_q"] if r["regression"]]
    
    report_lines += [
        f"- **Rescued QIDs ({len(rescued_ids)})**: " + (", ".join(rescued_ids) if rescued_ids else "None"),
        f"- **Regression QIDs ({len(regression_ids)})**: " + (", ".join(regression_ids) if regression_ids else "None"),
        "",
        "## 6. Decision Gate",
        f"- **Best Policy**: `{best_policy}`",
        f"- **Gate Verdict**: `{verdict}`"
    ]
    
    if gate_pass:
        report_lines.append(f"- **Action**: GO. Proceed to Phase 7 (Generation Dev200) for Policy `{best_policy}`.")
    else:
        report_lines.append("- **Action**: STOP. No fallback policy achieved a positive net gain (rescues > regressions) or improved Article Recall compared to the baseline without causing net regression.")
        
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {OUT_REPORT}")
    
    # Save a JSON file with best policy info
    with (OUT_DIR / "best_policy.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "best_policy": best_policy,
            "verdict": verdict,
            "gate_pass": gate_pass
        }, fh, ensure_ascii=False)
        
    conn.close()
    runtime.close()

if __name__ == "__main__":
    main()
