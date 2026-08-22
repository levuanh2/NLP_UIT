import time
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    from app.core.config import get_settings
    settings = get_settings()
    
    from app.retrieval.active_index import ActiveIndex
    active = ActiveIndex(settings.index_root_dir)
    
    conn = sqlite3.connect(active.bm25_path)
    
    query = "Thủ tục cấp giấy phép hoạt động đối với cơ sở dịch vụ thẩm mỹ?"
    terms = [token.replace('"', "") for token in query.split() if token.strip()]
    
    for num_terms in [3, 5, 8, 12, len(terms)]:
        sliced_terms = terms[:num_terms]
        match_query = " OR ".join(f'"{term}"' for term in sliced_terms)
        
        t0 = time.perf_counter()
        rows = conn.execute(
            "SELECT child_id, bm25(chunks) AS score FROM chunks "
            "WHERE chunks MATCH ? ORDER BY score, child_id LIMIT 20",
            (match_query,),
        ).fetchall()
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"Terms count: {num_terms:<2} | Query: {match_query[:50]}... | Latency: {dt:.1f}ms | Results: {len(rows)}")

    conn.close()

if __name__ == "__main__":
    main()
