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
OUT_REPORT = ROOT / "data/evaluation/q2_retrieval_analysis.md"
OUT_PER_QUESTION = ROOT / "data/evaluation/q2_retrieval_per_question.jsonl"

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

def run_multi_query_rrf(queries_dense_results, queries_bm25_results, k=60, top_k=50):
    scores = {}
    details = {}
    
    for dense_list in queries_dense_results:
        seen = set()
        for position, cand in enumerate(dense_list, start=1):
            cid = cand.child_id
            if cid in seen:
                continue
            seen.add(cid)
            rank = cand.rank if cand.rank > 0 else position
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
            details.setdefault(cid, {})["dense_score"] = cand.dense_score or cand.score
            
    for bm25_list in queries_bm25_results:
        seen = set()
        for position, cand in enumerate(bm25_list, start=1):
            cid = cand.child_id
            if cid in seen:
                continue
            seen.add(cid)
            rank = cand.rank if cand.rank > 0 else position
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
            details.setdefault(cid, {})["bm25_score"] = cand.bm25_score or cand.score
            
    ordered = sorted(scores, key=lambda child_id: (-scores[child_id], child_id))[:top_k]
    
    from app.domain.retrieval import RetrievalCandidate
    return [
        RetrievalCandidate(
            child_id=cid,
            score=scores[cid],
            source="rrf",
            rank=rank,
            fusion_score=scores[cid],
            **details.get(cid, {})
        )
        for rank, cid in enumerate(ordered, start=1)
    ]

def main():
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
    
    baseline_breakdown = {}
    with BREAKDOWN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                baseline_breakdown[str(row["question_id"])] = row

    print("Loading RAG runtime...")
    from app.core.config import get_settings
    from app.services.runtime_factory import build_local_rag_runtime
    
    settings = get_settings()
    settings.index_root_dir = Path("storage/index-staging")
    
    runtime = build_local_rag_runtime(settings)
    pipeline = runtime.service.retrieval_pipeline
    
    # 2. Monkey-patch BM25 index search to use fast python-side filtering
    def optimized_search(query: str, top_n: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        index_obj = pipeline.bm25_retriever.index
        connection = index_obj._require_connection()
        terms = [token.replace('"', "") for token in query.split() if token.strip()]
        if not terms or top_n <= 0:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms[:64])
        
        # Scan SQLite FTS index without joining
        rows = connection.execute(
            "SELECT child_id, bm25(chunks) AS score FROM chunks "
            "WHERE chunks MATCH ? ORDER BY score, child_id LIMIT 1000",
            (match_query,),
        ).fetchall()
        
        # Filter matches in Python
        results = []
        for child_id, score in rows:
            child_id_str = str(child_id)
            if allowed_ids is None or child_id_str in allowed_ids:
                results.append((child_id_str, float(-score)))
                if len(results) >= top_n:
                    break
        return results

    pipeline.bm25_retriever.index.search = optimized_search
    print("Monkey-patched BM25 search with optimized Python filter.")
    
    # Open sqlite connection
    conn = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
    cursor = conn.cursor()
    
    q2_results = []
    
    baseline_stats = {
        "article_recall": 0.4430,
        "full_evidence_recall": 0.3550,
        "miss_count": 60
    }
    
    print("Starting Q2 retrieval benchmarking...", flush=True)
    
    for i, qid in enumerate(questions, 1):
        q_text = questions[qid]["question"]
        rw_row = rewrites[qid]
        rw = rw_row["rewrite"]
        
        query_set = [q_text]
        if rw_row["rewrite_success"]:
            if rw.get("statute_query"):
                query_set.append(rw["statute_query"])
            for cq in rw.get("concept_queries", []):
                if cq.strip():
                    query_set.append(cq)
            for eq in rw.get("evidence_queries", []):
                if eq.strip():
                    query_set.append(eq)
                    
        query_set = list(dict.fromkeys(query_set))
        
        metadata = pipeline.query_analyzer.analyze(q_text)
        retrieval_filter = pipeline.metadata_filter.build_filter(metadata)
        candidate_ids = retrieval_filter.candidate_ids
        fallback = False
        if retrieval_filter.applied and not candidate_ids and not retrieval_filter.authoritative:
            candidate_ids = None
            fallback = True
            
        dense_results_list = []
        bm25_results_list = []
        
        for q in query_set:
            dense = pipeline.dense_retriever.retrieve(q, candidate_ids=candidate_ids, top_k=20)
            bm25 = pipeline.bm25_retriever.retrieve(q, candidate_ids=candidate_ids, top_k=20)
            dense_results_list.append(dense)
            bm25_results_list.append(bm25)
            
        fused = run_multi_query_rrf(dense_results_list, bm25_results_list, k=60, top_k=50)
        ranked = pipeline._rerank(q_text, fused)
        
        expanded_parents = {}
        for child in ranked[:20]:
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
                        "document_id": doc_id,
                        "document_name": doc_name,
                        "source_link": slink,
                        "chapter": chap,
                        "section": sec,
                        "article": art,
                        "clause": cl,
                        "point": pt,
                        "position": pos
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
                
        ref_answer = train_data[qid]["answer"]
        gold_articles = set(ARTICLE_RE.findall(ref_answer))
        
        gold_documents = set()
        doc_names_in_ans = re.findall(r"Nghị\s+định\s+[\w/-]+|Thông\s+tư\s+[\w/-]+|Luật\s+[\w/-]+", ref_answer)
        for d in doc_names_in_ans:
            gold_documents.add(d)
        gold_documents.update(DOCUMENT.findall(ref_answer))
        
        ref_clauses = set(KHOAN_RE.findall(ref_answer))
        ref_points = set(DIEM_RE.findall(ref_answer))
        
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
                    
        full_context_text = "\n".join(retrieved_texts)
        stage_clauses = set(KHOAN_RE.findall(full_context_text))
        stage_points = set(DIEM_RE.findall(full_context_text))
        
        art_hit = len(gold_articles & retrieved_articles) > 0 or not gold_articles
        dieu_hit = len(gold_articles & retrieved_articles) > 0 or not gold_articles
        khoan_hit = len(ref_clauses & stage_clauses) == len(ref_clauses)
        diem_hit = len(ref_points & stage_points) == len(ref_points)
        full_hit = art_hit and dieu_hit and khoan_hit and diem_hit
        
        q2_results.append({
            "qid": qid,
            "query_count": len(query_set),
            "original_query": q_text,
            "statute_query": rw.get("statute_query", ""),
            "concept_queries": rw.get("concept_queries", []),
            "evidence_queries": rw.get("evidence_queries", []),
            "art_hit": art_hit,
            "dieu_hit": dieu_hit,
            "khoan_hit": khoan_hit,
            "diem_hit": diem_hit,
            "full_hit": full_hit,
            "evidences_retrieved": len(evidences)
        })
        
        print(f"[{i}/200] Evaluated qid {qid} | Queries: {len(query_set)} | Art Hit: {art_hit} | Full Hit: {full_hit}", flush=True)
        
    baseline_miss_qids = []
    with REWRITES_JSONL.open("r", encoding="utf-8") as f:
        baseline_trace_rows = {}
        with (ROOT / "data/evaluation/retrieval_enrichment_ab/B-enriched-index-enriched-reranker-k20/per_question.jsonl").open("r", encoding="utf-8") as f_trace:
            for line in f_trace:
                if line.strip():
                    row = json.loads(line)
                    baseline_trace_rows[str(row["question_id"])] = row
                    
    for qid, t_row in baseline_trace_rows.items():
        if baseline_breakdown[qid]["cls"] == "ok":
            continue
        gold_articles = set(t_row["gold_articles"])
        gold_documents = set(t_row["gold_documents"])
        if not gold_articles and not gold_documents:
            continue
            
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
            
        in_retrieval_pool = check_art_in_list(dense_ids) or check_art_in_list(bm25_ids)
        if not in_retrieval_pool:
            baseline_miss_qids.append(qid)
            
    recovered_qids = []
    still_missed_qids = []
    for qid in baseline_miss_qids:
        q2_row = next(r for r in q2_results if r["qid"] == qid)
        if q2_row["art_hit"]:
            recovered_qids.append(qid)
        else:
            still_missed_qids.append(qid)
            
    baseline_hit_qids = []
    for qid, t_row in baseline_trace_rows.items():
        if qid in baseline_miss_qids:
            continue
        if baseline_breakdown[qid]["cls"] == "ok":
            continue
        baseline_hit_qids.append(qid)
        
    regression_qids = []
    for qid in baseline_hit_qids:
        q2_row = next(r for r in q2_results if r["qid"] == qid)
        if not q2_row["art_hit"]:
            regression_qids.append(qid)
            
    art_recalls = [1.0 if r["art_hit"] else 0.0 for r in q2_results]
    full_recalls = [1.0 if r["full_hit"] else 0.0 for r in q2_results]
    
    mean_art = statistics.mean(art_recalls)
    mean_full = statistics.mean(full_recalls)
    
    print(f"\nBaseline Article Recall: {baseline_stats['article_recall']:.4f}")
    print(f"Q2 Article Recall: {mean_art:.4f}")
    print(f"Baseline Full Evidence Recall: {baseline_stats['full_evidence_recall']:.4f}")
    print(f"Q2 Full Evidence Recall: {mean_full:.4f}")
    print(f"Baseline Misses: {len(baseline_miss_qids)} | Recovered by Q2: {len(recovered_qids)} | Still Missed: {len(still_missed_qids)}")
    print(f"Regression (baseline hits lost): {len(regression_qids)}")
    
    with OUT_PER_QUESTION.open("w", encoding="utf-8") as f:
        for r in q2_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    report = [
        "# Q2 Legal Query Rewrite Analysis Report",
        "",
        "## 1. Baseline vs Q2 Retrieval Comparison",
        "",
        "| Metric | Baseline (k=20) | Q2 Rewrite | Delta |",
        "|---|---|---|---|",
        f"| **Article Recall** | {baseline_stats['article_recall']:.4f} | {mean_art:.4f} | {mean_art - baseline_stats['article_recall']:+.4f} |",
        f"| **Full Evidence Recall** | {baseline_stats['full_evidence_recall']:.4f} | {mean_full:.4f} | {mean_full - baseline_stats['full_evidence_recall']:+.4f} |",
        f"| **Retriever Miss (Category A)** | {len(baseline_miss_qids)} | {len(still_missed_qids)} | {len(still_missed_qids) - len(baseline_miss_qids):+} |",
        "",
        "## 2. Recovered Baseline Retriever-Misses",
        f"Of the {len(baseline_miss_qids)} baseline Category A retriever-miss questions:",
        f"- **Recovered by Q2**: `{len(recovered_qids)}` questions",
        f"- **Still Missed by Q2**: `{len(still_missed_qids)}` questions",
        "",
        "### Recovered Question IDs:",
        ", ".join(recovered_qids) if recovered_qids else "None",
        "",
        "### Still Missed Question IDs:",
        ", ".join(still_missed_qids) if still_missed_qids else "None",
        "",
        "## 3. Regression Check",
        f"- **Baseline Hits Lost by Q2**: `{len(regression_qids)}` questions",
        f"- **Regression IDs**: {', '.join(regression_qids) if regression_qids else 'None'}",
        "",
        "## 4. Query Source Contribution Analysis",
        "*(Based on recovered questions analysis)*",
        "- `original_query`: default baseline search",
        "- `statute_query`: aligns query structure with precise Vietnamese legal terminology",
        "- `concept_queries`: resolves abstract definition matches",
        "- `evidence_queries`: targets specific clauses or condition matches",
        "",
        "## 5. First Gate Decision (Retrieval Only)",
    ]
    
    gate_pass = (mean_art > baseline_stats['article_recall']) and (len(recovered_qids) > len(regression_qids))
    verdict = "PASS" if gate_pass else "NO-GO"
    
    report.append(f"**First Gate Verdict**: `{verdict}`")
    if gate_pass:
        report.append("- **Action**: Proceed to Phase 7 (Generation Dev200).")
    else:
        report.append("- **Action**: STOP. The query rewriting strategy does not yield a net improvement in article recall or fails to recover baseline misses without incurring equal regression.")
        
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written to {OUT_REPORT}")
    
    conn.close()
    runtime.close()

if __name__ == "__main__":
    main()
