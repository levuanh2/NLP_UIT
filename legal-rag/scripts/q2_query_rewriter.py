import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_JSON = ROOT / "data/questions/dev200.json"
CACHE_FILE = ROOT / "data/evaluation/q2_query_rewrites.jsonl"

REWRITER_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Hãy chuyển đổi câu hỏi pháp lý của người dùng thành các câu truy vấn (queries) tối ưu cho công cụ tìm kiếm tài liệu luật (Dense/BM25 retrieval).

YÊU CẦU:
1. Xác định vấn đề pháp lý cốt lõi (legal_issue).
2. Tạo 1 truy vấn viết theo văn phong điều luật (statute_query) - dùng các thuật ngữ chính xác trong văn bản pháp luật thay vì ngôn ngữ tự nhiên.
3. Tạo 2-3 truy vấn khái niệm (concept_queries) - tập trung vào các từ khóa, thuật ngữ pháp lý cốt lõi.
4. Tạo 1-2 truy vấn bằng chứng (evidence_queries) - tập trung vào điều kiện, ngoại lệ, hoặc hành vi vi phạm cụ thể.

QUY TẮC QUAN TRỌNG:
- TUYỆT ĐỐI KHÔNG tự bịa ra hay đoán số Điều, số Khoản, số Điểm (ví dụ: cấm đoán "Điều 12", "Khoản 3", "Điểm a") nếu câu hỏi gốc không tự cung cấp thông tin đó. Chỉ trích xuất khái niệm pháp lý.
- Output phải là định dạng JSON duy nhất, không thêm lời giải thích nào khác.

ĐỊNH DẠNG OUTPUT JSON:
{{
  "original_query": "câu hỏi gốc của người dùng",
  "legal_issue": "vấn đề pháp lý chính",
  "statute_query": "truy vấn theo văn phong điều luật",
  "concept_queries": [
    "truy vấn khái niệm 1",
    "truy vấn khái niệm 2"
  ],
  "evidence_queries": [
    "truy vấn bằng chứng 1",
    "truy vấn bằng chứng 2"
  ]
}}

CÂU HỎI NGƯỜI DÙNG:
{original_question}
"""

def clean_and_parse_json(text: str) -> dict | None:
    # Try regex match for JSON block
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    json_str = match.group(1)
    try:
        data = json.loads(json_str)
        required_keys = ["original_query", "legal_issue", "statute_query", "concept_queries", "evidence_queries"]
        if all(k in data for k in required_keys):
            return data
    except Exception:
        pass
    return None

def main():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    questions = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    
    # Overwrite the cache to regenerate all 200 properly without LoRA adapter
    print(f"Clearing previous cache at {CACHE_FILE}...")
    if CACHE_FILE.is_file():
        CACHE_FILE.unlink()
        
    print("Loading base LLM (without LoRA adapter) for query rewriting...")
    from app.core.config import get_settings
    from app.generation.llm.factory import LLMGeneratorFactory
    
    settings = get_settings()
    settings.index_root_dir = Path("storage/index-staging")
    
    # Crucial: Disable adapter path to load raw base model
    llm = LLMGeneratorFactory.create(
        provider="local_transformers",
        model_name=settings.llm_model_name,
        device=settings.reranker_device,
        dtype=settings.model_dtype,
        local_files_only=settings.model_local_files_only,
        trust_remote_code=settings.model_trust_remote_code,
        max_new_tokens=256,
        temperature=0.1,
        top_p=settings.top_p,
        do_sample=False,
        min_new_tokens=10,
        repetition_penalty=settings.repetition_penalty,
        quantization=settings.model_quantization,
        adapter_path="", # Load base model without LoRA
    )
    llm.load()
    
    import hashlib
    prompt_hash = hashlib.md5(REWRITER_PROMPT.encode("utf-8")).hexdigest()
    
    with CACHE_FILE.open("w", encoding="utf-8") as out_f:
        for i, qid in enumerate(questions, 1):
            q_text = questions[qid]["question"]
            prompt = REWRITER_PROMPT.format(original_question=q_text)
            
            t0 = time.perf_counter()
            output = llm.generate(prompt, max_new_tokens=256, temperature=0.1)
            dt = time.perf_counter() - t0
            
            rewrite = clean_and_parse_json(output)
            success = rewrite is not None
            fallback_used = False
            
            if not success:
                fallback_used = True
                rewrite = {
                    "original_query": q_text,
                    "legal_issue": "failed_to_parse",
                    "statute_query": q_text,
                    "concept_queries": [q_text],
                    "evidence_queries": [q_text]
                }
                
            payload = {
                "qid": qid,
                "original_query": q_text,
                "rewrite": rewrite,
                "rewriter_model": settings.llm_model_name,
                "prompt_hash": prompt_hash,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rewrite_success": success,
                "rewrite_failure": not success,
                "fallback_used": fallback_used
            }
            
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out_f.flush()
            
            print(f"[{i}/{len(questions)}] Rewrote qid {qid} in {dt:.1f}s | Success: {success} | Sample query: {rewrite.get('statute_query', '')[:40]}")
            
    llm.unload()
    print("Query rewriting phase completed.")

if __name__ == "__main__":
    main()
