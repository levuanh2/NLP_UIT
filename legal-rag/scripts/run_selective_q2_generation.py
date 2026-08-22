import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SQLITE = ROOT / "storage/indexes/v1/metadata/legal.sqlite"
QUESTIONS_JSON = ROOT / "data/questions/dev200.json"
TRAIN_JSON = ROOT / "data/train/train.json"
REWRITES_JSONL = ROOT / "data/evaluation/q2_query_rewrites.jsonl"
OUT_DIR = ROOT / "data/evaluation/selective_q2"
OUT_SUBMISSION = ROOT / "data/outputs/selective_q2/submission.json"

ARTICLE_RE = re.compile(r"Điều\s+(\d+)")

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

def main():
    best_policy_json = OUT_DIR / "best_policy.json"
    if not best_policy_json.is_file():
        print("Error: best_policy.json not found! Please run the benchmark script first.", file=sys.stderr)
        sys.exit(1)
        
    policy_info = json.loads(best_policy_json.read_text(encoding="utf-8"))
    if not policy_info.get("gate_pass"):
        print("Verdict is NO-GO. Stopping generation run as requested.")
        sys.exit(0)
        
    best_policy = policy_info["best_policy"]
    print(f"Starting Generation using best fallback policy: {best_policy}")
    
    # 1. Load rewrites
    rewrites = {}
    with REWRITES_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rewrites[row["qid"]] = row
                
    questions = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    
    # Build RAG runtime
    from app.core.config import get_settings
    from app.services.runtime_factory import build_local_rag_runtime
    
    settings = get_settings()
    settings.index_root_dir = Path("storage/index-staging")
    settings.llm_adapter_path = "models/qlora-lr5e4/checkpoint-350"
    settings.min_new_tokens = 500
    
    runtime = build_local_rag_runtime(settings)
    pipeline = runtime.service.retrieval_pipeline
    
    # 2. Get score distribution to define percentiles for triggers
    print("Pre-profiling baseline reranker scores for policy triggers...")
    baseline_scores = []
    
    conn = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
    cursor = conn.cursor()
    
    # Monkey-patch BM25 search
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
    
    for qid in questions:
        q_text = questions[qid]["question"]
        metadata = pipeline.query_analyzer.analyze(q_text)
        retrieval_filter = pipeline.metadata_filter.build_filter(metadata)
        candidate_ids = retrieval_filter.candidate_ids
        if retrieval_filter.applied and not candidate_ids and not retrieval_filter.authoritative:
            candidate_ids = None
            
        dense = pipeline.dense_retriever.retrieve(q_text, candidate_ids=candidate_ids, top_k=20)
        bm25 = pipeline.bm25_retriever.retrieve(q_text, candidate_ids=candidate_ids, top_k=20)
        fused = pipeline.fusion.fuse(dense, bm25, k=60, top_k=30)
        ranked = pipeline._rerank(q_text, fused)
        top_score = ranked[0].score if len(ranked) > 0 else 0.0
        baseline_scores.append(top_score)
        
    p10 = sorted(baseline_scores)[int(len(baseline_scores)*0.10)]
    p25 = sorted(baseline_scores)[int(len(baseline_scores)*0.25)]
    print(f"Percentiles loaded: P10={p10:.4f}, P25={p25:.4f}")
    
    # 3. Patch pipeline retrieve to implement selective query-rewrite fallback
    original_retrieve = pipeline.retrieve
    
    from app.domain.queries import LegalQuery
    from app.domain.retrieval import RetrievalResult
    
    # Keep track of active qid during runtime service answer
    # We can inspect the query string or look for qid from questions mapping
    q_to_qid = {questions[qid]["question"]: qid for qid in questions}
    
    def selective_retrieve(query: str | LegalQuery) -> RetrievalResult:
        raw_query = query.question if isinstance(query, LegalQuery) else query
        qid = q_to_qid.get(raw_query)
        
        # 1. Run baseline retrieve
        result = original_retrieve(query)
        
        if not qid or qid not in rewrites:
            return result
            
        # 2. Check trigger
        rw_row = rewrites[qid]
        rw = rw_row["rewrite"]
        
        top_score = result.candidates[0].score if len(result.candidates) > 0 else 0.0
        
        metadata = result.query_metadata
        doc_detected = bool(metadata.document_name or metadata.article)
        has_matching_candidate = False
        if doc_detected:
            for cand in result.candidates[:5]:
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
            
        triggered = False
        if best_policy == "F1":
            triggered = (doc_detected and not has_matching_candidate)
        elif best_policy == "F2":
            triggered = (top_score < 0.5)
        elif best_policy == "F3":
            triggered = (top_score < p25)
        elif best_policy == "F4":
            triggered = (doc_detected and not has_matching_candidate) or (top_score < p10)
            
        if triggered and rw_row["rewrite_success"]:
            print(f"Triggered fallback for qid {qid}")
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
            
            # Retrieve candidate_ids from original filter
            retrieval_filter = pipeline.metadata_filter.build_filter(metadata)
            candidate_ids = retrieval_filter.candidate_ids
            if retrieval_filter.applied and not candidate_ids and not retrieval_filter.authoritative:
                candidate_ids = None
                
            dense_lists = []
            bm25_lists = []
            for q in query_set:
                dense_lists.append(pipeline.dense_retriever.retrieve(q, candidate_ids=candidate_ids, top_k=20))
                bm25_lists.append(pipeline.bm25_retriever.retrieve(q, candidate_ids=candidate_ids, top_k=20))
                
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
            
            primary_ids = {c.child_id for c in result.candidates}
            merged_candidates = list(result.candidates)
            for c in rewrite_cands:
                if c.child_id not in primary_ids:
                    merged_candidates.append(c)
            
            # Rerank against original user query
            ranked = pipeline._rerank(raw_query, merged_candidates)
            expanded = pipeline.parent_expander.expand(ranked)
            context = pipeline.context_builder.build(raw_query, expanded)
            
            # Reconstruct RetrievalResult
            return RetrievalResult(
                query=raw_query,
                query_metadata=metadata,
                candidates=ranked,
                evidences=context.evidences,
                active_index_version=result.active_index_version,
                dense_count=result.dense_count,
                bm25_count=result.bm25_count,
                fused_count=result.fused_count,
                reranked_count=len(ranked),
                metadata_filter_applied=result.metadata_filter_applied,
                metadata_filter_fallback=result.metadata_filter_fallback
            )
        else:
            return result
            
    pipeline.retrieve = selective_retrieve
    print("Patched retrieval pipeline successfully with selective fallback.")
    
    # 4. Load references and question dataset
    from app.submission.question_loader import QuestionDatasetLoader
    dataset = QuestionDatasetLoader().load(QUESTIONS_JSON)
    
    # Run end-to-end generation
    OUT_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission: dict[str, dict] = {}
    
    t_total = time.perf_counter()
    for i, query in enumerate(dataset, 1):
        t0 = time.perf_counter()
        answer = runtime.service.answer(query)
        dt = time.perf_counter() - t0
        
        qid = query.question_id
        pred = answer.answer
        submission[qid] = {"answer": pred}
        print(f"[{i}/200] Generated qid {qid} in {dt:.1f}s")
        
    wall = time.perf_counter() - t_total
    print(f"Generation done in {wall:.0f}s")
    
    # Save submission
    OUT_SUBMISSION.write_text(json.dumps(submission, ensure_ascii=False, indent=4), encoding="utf-8")
    
    # Run scorer
    eval_cmd = [
        str(ROOT / ".venv/bin/python"),
        "scripts/eval_dev.py",
        "--score", str(OUT_SUBMISSION),
        "--train", str(TRAIN_JSON),
        "--questions", str(QUESTIONS_JSON),
        "--allow-failures",
    ]
    ev = subprocess.run(eval_cmd, cwd=ROOT, capture_output=True, text=True)
    print(ev.stdout)
    
    m_meteor = re.search(r"METEOR\s+([\d.]+)", ev.stdout)
    m_rouge = re.search(r"ROUGE-L\s+([\d.]+)", ev.stdout)
    
    meteor = float(m_meteor.group(1)) if m_meteor else 0.0
    rouge = float(m_rouge.group(1)) if m_rouge else 0.0
    
    delta = meteor - 0.4880
    passed_gate = delta >= 0.028
    verdict = "CANDIDATE WINNER" if passed_gate else "NO-GO"
    
    print(f"\nGeneration results for policy {best_policy}:")
    print(f"METEOR: {meteor:.4f} (baseline 0.4880 | delta {delta:+.4f})")
    print(f"ROUGE-L: {rouge:.4f} (baseline 0.3345)")
    print(f"Verdict: {verdict}")
    
    # Write details to the report markdown
    report_path = OUT_DIR / "selective_q2_report.md"
    report_text = report_path.read_text(encoding="utf-8")
    
    gen_section = [
        "",
        "## 7. End-to-end Generation Results",
        f"- **METEOR**: `{meteor:.4f}` (baseline 0.4880, delta `{delta:+.4f}`)",
        f"- **ROUGE-L**: `{rouge:.4f}` (baseline 0.3345)",
        f"- **Decision Gate Verdict**: `{verdict}`",
        f"- **Action**: " + ("GO. Presenting candidate for manual review!" if passed_gate else "NO-GO. Preserving v5 winner.")
    ]
    
    report_path.write_text(report_text + "\n".join(gen_section), encoding="utf-8")
    print(f"Report updated at {report_path}")
    
    conn.close()
    runtime.close()

if __name__ == "__main__":
    main()
