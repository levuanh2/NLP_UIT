import json
import re
import sqlite3
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_JSON = ROOT / "data/questions/dev200.json"
TRAIN_JSON = ROOT / "data/train/train.json"

DOCUMENT = re.compile(r"(\d+/\d{4}/[A-ZĐ][A-ZĐ-]*|\d+-\d{4}-[A-ZĐ][A-ZĐ-]*)$")
ARTICLE_RE = re.compile(r"Điều\s+(\d+)")

def normalize_doc_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFD", name.casefold())
    s = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    s = s.replace("đ", "d").replace("/", "-")
    s = re.sub(r"[^a-z0-9-]", "", s)
    return s

def main():
    questions = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    train_data = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    
    from app.core.config import get_settings
    from app.services.runtime_factory import build_local_rag_runtime
    
    settings = get_settings()
    settings.index_root_dir = Path("storage/index-staging")
    settings.llm_adapter_path = "models/qlora-lr5e4/checkpoint-350"
    
    settings.reranker_top_k = 20
    settings.dense_top_k = 20
    settings.bm25_top_k = 20
    settings.rrf_top_k = 30
    
    runtime = build_local_rag_runtime(settings)
    pipeline = runtime.service.retrieval_pipeline
    
    sqlite_path = pipeline.bm25_retriever.active_index.sqlite_path
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    
    for i, qid in enumerate(list(questions.keys())[:5], 1):
        q_text = questions[qid]["question"]
        ref_answer = train_data[qid]["answer"]
        
        metadata = pipeline.query_analyzer.analyze(q_text)
        retrieval_filter = pipeline.metadata_filter.build_filter(metadata)
        candidate_ids = retrieval_filter.candidate_ids
        if retrieval_filter.applied and not candidate_ids and not retrieval_filter.authoritative:
            candidate_ids = None
            
        dense = pipeline.dense_retriever.retrieve(q_text, candidate_ids=candidate_ids, top_k=20)
        bm25 = pipeline.bm25_retriever.retrieve(q_text, candidate_ids=candidate_ids, top_k=20)
        fused = pipeline.fusion.fuse(dense, bm25, k=60, top_k=30)
        ranked = pipeline._rerank(q_text, fused)
        
        gold_articles = set(ARTICLE_RE.findall(ref_answer))
        gold_documents = set()
        doc_names_in_ans = re.findall(r"Nghị\s+định\s+[\w/-]+|Thông\s+tư\s+[\w/-]+|Luật\s+[\w/-]+", ref_answer)
        for d in doc_names_in_ans:
            gold_documents.add(d)
        gold_documents.update(DOCUMENT.findall(ref_answer))
        
        # Parent expansion
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
                        "document_name": doc_name,
                        "article": art,
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
                
        retrieved_articles = set()
        for ev in evidences:
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
        
        print(f"--- QID {qid} ---")
        print(f"Gold articles: {gold_articles}")
        print(f"Gold documents: {gold_documents}")
        print(f"Evidences count: {len(evidences)}")
        for ev in evidences:
            print(f"  Ev doc: {ev['document_name']} | art: {ev['article']}")
        print(f"Retrieved articles (matching docs): {retrieved_articles}")
        print(f"Art hit: {art_hit}")
        
    conn.close()
    runtime.close()

if __name__ == "__main__":
    main()
