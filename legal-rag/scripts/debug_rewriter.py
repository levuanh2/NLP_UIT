import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_JSON = ROOT / "data/questions/dev200.json"

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

def main():
    questions = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    qid = list(questions.keys())[0]
    q_text = questions[qid]["question"]
    
    from app.core.config import get_settings
    from app.generation.llm.factory import LLMGeneratorFactory
    
    settings = get_settings()
    settings.index_root_dir = Path("storage/index-staging")
    settings.llm_adapter_path = "models/qlora-lr5e4/checkpoint-350"
    
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
        adapter_path=settings.llm_adapter_path,
    )
    llm.load()
    
    prompt = REWRITER_PROMPT.format(original_question=q_text)
    print("--- PROMPT ---")
    print(prompt)
    
    output = llm.generate(prompt, max_new_tokens=256, temperature=0.1)
    print("--- RAW OUTPUT ---")
    print(repr(output))
    print("------------------")
    
    llm.unload()

if __name__ == "__main__":
    main()
