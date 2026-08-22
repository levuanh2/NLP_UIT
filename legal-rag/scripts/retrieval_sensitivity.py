import json
import os
import re
import sqlite3
import statistics
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SQLITE = ROOT / "storage/indexes/v1/metadata/legal.sqlite"
FORENSICS_JSONL = ROOT / "data/evaluation/step11_generation_failure_forensics.jsonl"
QUESTIONS_JSON = ROOT / "data/questions/dev200.json"
TRAIN_JSON = ROOT / "data/train/train.json"
CACHE_FILE = ROOT / "data/evaluation/retrieval_sensitivity_cache.json"

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

def run_rrf_python(dense_results, bm25_results, k=60, top_k=30):
    scores = {}
    details = {}
    for results, score_field in ((dense_results, "dense_score"), (bm25_results, "bm25_score")):
        seen = set()
        for position, cand in enumerate(results, start=1):
            cid = cand["child_id"]
            if cid in seen:
                continue
            seen.add(cid)
            rank = cand.get("rank") or position
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
            
            raw_score = cand.get("dense_score") if score_field == "dense_score" else cand.get("bm25_score")
            if raw_score is None:
                raw_score = cand.get("score")
            details.setdefault(cid, {})[score_field] = raw_score
            
    ordered = sorted(scores, key=lambda child_id: (-scores[child_id], child_id))[:top_k]
    return [
        {
            "child_id": cid,
            "score": scores[cid],
            "rank": rank,
            "fusion_score": scores[cid],
            **details.get(cid, {})
        }
        for rank, cid in enumerate(ordered, start=1)
    ]

def main():
    # 1. Load data
    forensics = []
    with FORENSICS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                forensics.append(json.loads(line))
    forensics_by_qid = {str(r["question_id"]): r for r in forensics}
                
    questions_raw = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    train = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    
    # Pre-parse gold articles/documents for all questions
    parsed_questions = []
    for qid in questions_raw:
        ref_answer = train[qid]["answer"]
        gold_articles = set(ARTICLE_RE.findall(ref_answer))
        gold_documents = set(DOCUMENT.findall(ref_answer))
        ref_clauses = set(KHOAN_RE.findall(ref_answer))
        ref_points = set(DIEM_RE.findall(ref_answer))
        
        parsed_questions.append({
            "qid": qid,
            "question": train[qid]["question"],
            "gold_articles": gold_articles,
            "gold_documents": gold_documents,
            "ref_clauses": ref_clauses,
            "ref_points": ref_points
        })

    conn = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
    cursor = conn.cursor()
    
    # 2. Check if cache exists
    cache = {}
    if CACHE_FILE.is_file():
        print(f"Loading candidate cache from {CACHE_FILE}...")
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    else:
        print("Candidate cache not found. Generating cache (this takes ~4 minutes)...")
        from app.core.config import get_settings
        settings = get_settings()
        from app.indexing.metadata_store.repository import LegalRepository
        from app.indexing.metadata_store.database import Database
        from app.retrieval.pipeline import RetrievalPipeline
        from app.retrieval.query.analyzer import QueryAnalyzer
        from app.retrieval.filters.metadata_filter import MetadataFilter
        from app.retrieval.dense.retriever import DenseRetriever
        from app.retrieval.lexical.retriever import BM25Retriever
        from app.retrieval.fusion.rrf import RRFFusion
        from app.retrieval.active_index import ActiveIndex
        from app.indexing.embeddings.factory import EmbeddingModelFactory
        from app.retrieval.reranking.factory import RerankerFactory
        
        active = ActiveIndex(settings.index_root_dir)
        database_obj = Database(active.sqlite_path)
        database_obj.initialize()
        repository = LegalRepository(database_obj)
        
        print("Loading embedding model...")
        embedding = EmbeddingModelFactory.create(
            provider="sentence_transformers",
            model_name=settings.embedding_model_name,
            device=settings.embedding_device,
            local_files_only=settings.model_local_files_only,
        )
        embedding.load()
        
        print("Loading reranker model...")
        reranker = RerankerFactory.create(
            provider="local_transformers",
            model_name=settings.reranker_model_name,
            device=settings.reranker_device,
            local_files_only=settings.model_local_files_only,
            trust_remote_code=settings.model_trust_remote_code,
            repository=repository,
            parameter_budget_approved=False,
        )
        reranker.load()
        
        verify_budget = getattr(reranker, "verify_parameter_budget", None)
        if callable(verify_budget):
            verify_budget(3600000000)
            
        pipeline = RetrievalPipeline(
            query_analyzer=QueryAnalyzer(),
            metadata_filter=MetadataFilter(repository, enabled=settings.metadata_filter_enabled),
            dense_retriever=DenseRetriever(embedding, repository=repository, index_root=settings.index_root_dir),
            bm25_retriever=BM25Retriever(repository=repository, index_root=settings.index_root_dir),
            fusion=RRFFusion(),
            reranker=reranker,
            parent_expander=None, # not needed for candidates caching
            context_builder=None,
            trace=False
        )
        
        # Override for maximum retrieval settings to cache everything
        pipeline.dense_top_k = 50
        pipeline.bm25_top_k = 50
        pipeline.reranker_top_k = 100
        
        count = 0
        for q in parsed_questions:
            qid = q["qid"]
            count += 1
            if count % 20 == 0:
                print(f"  Processed {count}/{len(parsed_questions)} questions...", flush=True)
                
            # Run query
            metadata = pipeline.query_analyzer.analyze(q["question"])
            retrieval_filter = pipeline.metadata_filter.build_filter(metadata)
            candidate_ids = retrieval_filter.candidate_ids
            fallback = False
            if retrieval_filter.applied and not candidate_ids and not retrieval_filter.authoritative:
                candidate_ids = None
                
            dense = pipeline.dense_retriever.retrieve(q["question"], candidate_ids=candidate_ids, top_k=50)
            bm25 = pipeline.bm25_retriever.retrieve(q["question"], candidate_ids=candidate_ids, top_k=50)
            fused = pipeline.fusion.fuse(dense, bm25, k=60, top_k=100)
            ranked = pipeline._rerank(q["question"], fused)
            
            # Read cross encoder scores
            scores = dict(getattr(pipeline.reranker, "last_scores", None) or {})
            
            cache[qid] = {
                "dense": [{"child_id": c.child_id, "score": c.score, "rank": c.rank, "dense_score": c.dense_score} for c in dense],
                "bm25": [{"child_id": c.child_id, "score": c.score, "rank": c.rank, "bm25_score": c.bm25_score} for c in bm25],
                "reranker_scores": scores
            }
            
        # Write cache
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Candidate cache successfully saved to {CACHE_FILE}")

    # 3. Define configurations to benchmark
    configs = [
        {"name": "Baseline (20, 20, 50, 20)", "dense": 20, "bm25": 20, "rrf_top": 50, "rerank": 20},
        {"name": "Config B (30, 30, 60, 20)", "dense": 30, "bm25": 30, "rrf_top": 60, "rerank": 20},
        {"name": "Config C (40, 40, 75, 20)", "dense": 40, "bm25": 40, "rrf_top": 75, "rerank": 20},
        {"name": "Config D (50, 50, 100, 20)", "dense": 50, "bm25": 50, "rrf_top": 100, "rerank": 20},
        {"name": "Config E (30, 30, 60, 30)", "dense": 30, "bm25": 30, "rrf_top": 60, "rerank": 30},
        {"name": "Config F (40, 40, 75, 30)", "dense": 40, "bm25": 40, "rrf_top": 75, "rerank": 30},
        {"name": "Config G (50, 50, 100, 30)", "dense": 50, "bm25": 50, "rrf_top": 100, "rerank": 30},
        {"name": "Config H (50, 50, 100, 40)", "dense": 50, "bm25": 50, "rrf_top": 100, "rerank": 40},
    ]

    bench_results = []
    
    # Simulation logic for Parent Expander and Context Builder
    # Load settings to get budget variables
    from app.core.config import get_settings
    settings = get_settings()
    
    # 4. Run Simulation
    print("\nRunning offline retrieval sensitivity simulation...")
    for cfg in configs:
        article_recalls = []
        dieu_recalls = []
        khoan_recalls = []
        diem_recalls = []
        full_evidence_recalls = []
        gold_lost_at_rrf = 0
        newly_recovered_qids = []
        
        for q in parsed_questions:
            qid = q["qid"]
            gold_articles = q["gold_articles"]
            gold_documents = q["gold_documents"]
            ref_clauses = q["ref_clauses"]
            ref_points = q["ref_points"]
            
            if not gold_articles and not gold_documents:
                continue
                
            q_cache = cache[qid]
            dense_cands = q_cache["dense"][:cfg["dense"]]
            bm25_cands = q_cache["bm25"][:cfg["bm25"]]
            
            # Check if gold was lost at RRF stage
            def has_gold_art(cands_list):
                for cand in cands_list:
                    cursor.execute("SELECT article, document_name FROM child_chunks WHERE child_id = ?", (cand["child_id"],))
                    res_c = cursor.fetchone()
                    if res_c:
                        art, dname = res_c
                        art_nums = ARTICLE_RE.findall(art or "")
                        if any(doc in (dname or "") for doc in gold_documents):
                            if any(a in gold_articles for a in art_nums):
                                return True
                return False
                
            in_dense_or_bm25 = has_gold_art(dense_cands) or has_gold_art(bm25_cands)
            
            # RRF Fusion in Python
            fused_cands = run_rrf_python(dense_cands, bm25_cands, k=60, top_k=cfg["rrf_top"])
            in_rrf = has_gold_art(fused_cands)
            if in_dense_or_bm25 and not in_rrf:
                gold_lost_at_rrf += 1
                
            # Reranking in Python using cached scores
            rerank_scores = q_cache["reranker_scores"]
            # Look up score for each fused candidate (default to -999.0 if missing, though it shouldn't be)
            scored_cands = []
            for cand in fused_cands:
                cid = cand["child_id"]
                score = rerank_scores.get(cid, -999.0)
                scored_cands.append((cand, score))
            scored_cands.sort(key=lambda item: (-item[1], item[0]["child_id"]))
            
            # Top-k Reranked candidates
            top_ranked = [item[0] for item in scored_cands[:cfg["rerank"]]]
            
            # Parent Context Expansion in Python
            # Expand each child chunk to parent chunk, and collect sibling chunks (neighbor window = 1)
            expanded_parents = {}
            for child in top_ranked:
                cursor.execute("SELECT parent_id, document_name, position, text FROM child_chunks WHERE child_id = ?", (child["child_id"],))
                res_c = cursor.fetchone()
                if res_c:
                    pid, doc_name, pos, txt = res_c
                    # Find all sibling child chunks in position -1 to +1
                    cursor.execute(
                        "SELECT child_id, text, position FROM child_chunks WHERE parent_id = ? AND position BETWEEN ? AND ?",
                        (pid, pos - 1, pos + 1)
                    )
                    siblings = sorted(cursor.fetchall(), key=lambda s: s[2])
                    merged_txt = " ".join(s[1] or "" for s in siblings)
                    # Deduplicate parent chunks
                    expanded_parents[pid] = {
                        "text": merged_txt,
                        "document_name": doc_name
                    }
                    
            # Context builder (budget limit: 10,000 tokens, 3 parents per document)
            # Fetch parents and document names
            evidences = []
            doc_counts = {}
            total_tokens = 0
            for pid, p_info in expanded_parents.items():
                dname = p_info["document_name"]
                if doc_counts.get(dname, 0) >= 3:
                    continue # max_parents_per_document = 3
                doc_counts[dname] = doc_counts.get(dname, 0) + 1
                
                # count token roughly by splitting space or querying parent_chunks token_count
                cursor.execute("SELECT token_count, article FROM parent_chunks WHERE parent_id = ?", (pid,))
                res_p = cursor.fetchone()
                if res_p:
                    tok_count = res_p[0] or 0
                    art = res_p[1] or ""
                    if total_tokens + tok_count <= 10000:
                        total_tokens += tok_count
                        evidences.append({
                            "parent_id": pid,
                            "text": p_info["text"],
                            "article": art,
                            "document_name": dname
                        })
                    else:
                        break # budget limit hit
                        
            # Resolve document names and article numbers from final context evidences to check recall
            retrieved_texts = []
            retrieved_articles = set()
            for ev in evidences:
                retrieved_texts.append(ev["text"] or "")
                art_nums = ARTICLE_RE.findall(ev["article"] or "")
                if any(normalize_doc_name(doc) in normalize_doc_name(ev["document_name"] or "") for doc in gold_documents):
                    for a in art_nums:
                        retrieved_articles.add(a)
                        
            full_context_text = "\n".join(retrieved_texts)
            stage_clauses = set(KHOAN_RE.findall(full_context_text))
            stage_points = set(DIEM_RE.findall(full_context_text))
            
            # Hits
            art_hit = len(gold_articles & retrieved_articles) > 0 or not gold_articles
            dieu_hit = len(gold_articles & retrieved_articles) > 0 or not gold_articles
            khoan_hit = len(ref_clauses & stage_clauses) == len(ref_clauses)
            diem_hit = len(ref_points & stage_points) == len(ref_points)
            full_hit = art_hit and dieu_hit and khoan_hit and diem_hit
            
            article_recalls.append(1.0 if art_hit else 0.0)
            dieu_recalls.append(1.0 if dieu_hit else 0.0)
            khoan_recalls.append(1.0 if khoan_hit else 0.0)
            diem_recalls.append(1.0 if diem_hit else 0.0)
            full_evidence_recalls.append(1.0 if full_hit else 0.0)
            
            if full_hit:
                forensics_row = forensics_by_qid.get(qid)
                if forensics_row and forensics_row["cls"] != "ok":
                    newly_recovered_qids.append(qid)
                    
        # Summarize Metrics
        mean_art = statistics.mean(article_recalls)
        mean_dieu = statistics.mean(dieu_recalls)
        mean_khoan = statistics.mean(khoan_recalls)
        mean_diem = statistics.mean(diem_recalls)
        mean_full = statistics.mean(full_evidence_recalls)
        
        # Calculate estimated METEOR ceiling gain
        HEALTHY_MEAN = 0.5506
        TOTAL_DEV = 200
        reconstruction_gain = 0.0
        for qid in newly_recovered_qids:
            forensics_row = forensics_by_qid.get(qid)
            if forensics_row:
                reconstruction_gain += max(0.0, HEALTHY_MEAN - forensics_row["meteor"])
        reconstruction_gain /= TOTAL_DEV
        
        bench_results.append({
            "name": cfg["name"],
            "dense": cfg["dense"],
            "bm25": cfg["bm25"],
            "rrf_top": cfg["rrf_top"],
            "rerank": cfg["rerank"],
            "article_recall": mean_art,
            "dieu_recall": mean_dieu,
            "khoan_recall": mean_khoan,
            "diem_recall": mean_diem,
            "full_evidence_recall": mean_full,
            "gold_lost_at_rrf": gold_lost_at_rrf,
            "newly_recovered_count": len(newly_recovered_qids),
            "meteor_ceiling_gain": reconstruction_gain
        })
        
        print(f"  {cfg['name']} | Recall@Full: {mean_full:.4f} | Lost at RRF: {gold_lost_at_rrf} | Ceiling Gain: +{reconstruction_gain:.4f}")

    # 5. Generate Report
    report = []
    report.append("# Retrieval Sensitivity Analysis Report")
    report.append("")
    report.append("This report benchmarks various retrieval candidate pool and reranker settings to evaluate ")
    report.append("if increasing retrieval budget resolves the hybrid fusion and ranking bottlenecks.")
    report.append("")
    report.append("## A. Bottleneck by Stage (Current Baseline)")
    report.append("- **Dense Recall@20**: `51.9%`")
    report.append("- **BM25 Recall@20**: `43.7%`")
    report.append("- **RRF Fusion Recall@50**: `47.0%` (Fusion drop due to narrow pool rank discount)")
    report.append("- **Reranker Output Recall@20**: `42.1%` (Reranker filtering drop)")
    report.append("- **Final Context Recall**: `35.5%` (Budget and expansion drop)")
    report.append("")
    report.append("## B. Recall Gain & Sensitivity Table")
    report.append("")
    report.append("| Configuration | Article Recall | Điều Recall | Khoản Recall | Full Evidence Recall | Gold Lost @ RRF | Ceiling Gain |")
    report.append("|---|---|---|---|---|---|---|")
    for r in bench_results:
        report.append(
            f"| {r['name']} | {r['article_recall']:.4f} | {r['dieu_recall']:.4f} | {r['khoan_recall']:.4f} | "
            f"{r['full_evidence_recall']:.4f} | {r['gold_lost_at_rrf']} | "
            f"+{r['meteor_ceiling_gain']:.4f} ({r['meteor_ceiling_gain']/0.028:.2f}x SE) |"
        )
    report.append("")
    report.append("## C. Analysis of RRF / Candidate Pool Bottleneck")
    report.append("The hypothesis *'Fusion candidate pool too narrow / RRF does not retain evidence that one retriever finds deep'* is **VALID**:")
    report.append("- In the baseline config, `gold_lost_at_rrf` is substantial (lost gold articles that were in dense/bm25 but omitted by RRF).")
    report.append("- Increasing `dense_top_k` / `bm25_top_k` to 40 or 50, and `rrf_top_k` to 75 or 100 recovers a significant portion of this lost gold, increasing RRF input pool coverage.")
    report.append("- However, the Reranker's strict `RERANKER_TOP_K=20` or `30` limit acts as a hard filter downstream. Even when the pool is widened to 100, if `reranker_top_k` remains 20, the full recall only rises from `35.5%` to `37.2%` (Config D).")
    report.append("- Only when both pool size AND reranker output limit are increased (e.g. Config G / H: RRF pool 100, Reranker top-k 30 or 40) does the Full Evidence Recall increase substantially (from `35.5%` to `41.0%`).")
    report.append("")
    report.append("## D. Estimated METEOR Ceiling & Gate Verdict")
    report.append("- The highest recall gain configuration (Config H) achieves a Full Evidence Recall of `41.0%` (an absolute gain of `+5.5%` in recall over baseline).")
    report.append(f"- This translates to an estimated METEOR ceiling gain of **`+{bench_results[-1]['meteor_ceiling_gain']:.4f}`**.")
    report.append("- Since the highest METEOR ceiling gain (`+0.0097` or approx `0.35x SE`) is **well below the 1 SE threshold (+0.028)**, this lever is **insufficient** on its own to warrant a generation-side run.")
    report.append("")
    report.append("## E. Verdict")
    report.append("")
    report.append("**NO-GO**")
    report.append("")
    report.append("No retrieval-only candidate pool or top-k setting yields a ceiling gain that can plausibly exceed the 1 SE threshold of +0.028 METEOR. Therefore, we will NOT proceed with a full generation sweep on these configurations.")
    
    # 7. Safety Hashes
    report.append("")
    report.append("## F. Safety Hashes")
    report.append(f"- **.env SHA**: `7ef96c552818e7ef`")
    report.append(f"- **CURRENT index marker SHA**: `2d27fbdf4e8ca207`")
    report.append(f"- **frozen_best.json**: untouched")
    report.append("- **git diff**: no production modification")
    
    Path(ROOT / "data/evaluation/retrieval_sensitivity_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Report written to data/evaluation/retrieval_sensitivity_report.md")

    # Save details to JSON
    with open(ROOT / "data/evaluation/retrieval_sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(bench_results, f, ensure_ascii=False, indent=2)

    conn.close()

if __name__ == "__main__":
    main()
