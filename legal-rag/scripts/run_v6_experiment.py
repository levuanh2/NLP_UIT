"""V6 End-to-End Experiment: Wider Candidate Pool + Evidence-Identification Prompt.

Changes from frozen v5 baseline:
  1. dense_top_k = 50  (was 20)
  2. bm25_top_k  = 50  (was 20)
  3. rrf_top_k   = 100 (was 30)
  4. reranker_top_k = 40 (was 20)
  5. System prompt adds internal evidence-identification step

Everything else (checkpoint, LoRA, index, context builder, evaluation) is identical.
No oracle / gold forcing — this is a production-feasible configuration.
"""

import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_JSON = ROOT / "data/questions/dev200.json"
TRAIN_JSON = ROOT / "data/train/train.json"
FORENSICS_JSONL = ROOT / "data/evaluation/step11_generation_failure_forensics.jsonl"
BREAKDOWN_JSONL = ROOT / "data/evaluation/step9_failure_breakdown.jsonl"

OUT_DIR = ROOT / "data/outputs/v6_experiment"
V6_CONFIG_JSON = ROOT / "data/evaluation/v6_config.json"
V6_PER_Q_JSONL = ROOT / "data/evaluation/v6_per_question.jsonl"
REPORT_MD = ROOT / "data/evaluation/v6_end_to_end_report.md"

ARTICLE_RE = re.compile(r"Điều\s+(\d+)")

# ── V6 System Prompt ──────────────────────────────────────────────────────────
V6_SYSTEM_PROMPT = """SYSTEM:
Bạn là chuyên gia pháp luật Việt Nam, soạn câu trả lời chuẩn cho người hỏi.

QUY TRÌNH NỘI BỘ (KHÔNG HIỂN THỊ RA):
Trước khi viết câu trả lời, hãy xác định chính xác trong ngữ cảnh:
- Điều (article) nào chứa câu trả lời trực tiếp.
- Khoản (clause) nào quy định chi tiết.
- Điểm (point) nào (nếu có) hỗ trợ trực tiếp.
Dùng các thông tin đã xác định này làm bằng chứng cốt lõi. Tuyệt đối không
hiển thị bước xác định này ra ngoài.

CẤU TRÚC BẮT BUỘC:
1. Mở đầu nêu căn cứ: "Căn cứ Điều ... Khoản ... của <tên văn bản>" hoặc
   "Theo quy định tại Điều ... <tên văn bản>", rồi "quy định như sau:".
2. Trích NGUYÊN VĂN nội dung điều khoản trong Ngữ cảnh. Giữ nguyên câu chữ,
   giữ nguyên cách đánh số a) b) c), 1. 2. 3. và gạch đầu dòng của văn bản gốc.
   Không diễn giải lại, không rút gọn, không thay từ đồng nghĩa.
3. Nếu có nhiều điều khoản liên quan, trích lần lượt từng điều, mỗi điều nêu rõ
   căn cứ trước khi trích.
4. Kết bằng câu chốt "Theo đó, ..." nhắc lại đúng phần quy định trả lời thẳng
   vào câu hỏi, dùng lại nguyên văn từ ngữ của điều khoản.

ĐỘ DÀI:
- Viết khoảng 350 đến 450 từ. Đây là yêu cầu bắt buộc, không được ngắn hơn.
- Trả lời ngắn bị chấm điểm rất thấp. Thà trích thừa điều khoản liên quan còn
  hơn thiếu. Khai thác hết nội dung Ngữ cảnh có liên quan đến câu hỏi.

QUY TẮC:
- Luôn đưa ra câu trả lời thực chất. Tuyệt đối không viết "không có thông tin",
  "không đủ căn cứ", "không tìm thấy", "không nêu rõ", "tài liệu chưa đủ" hay
  bất kỳ câu nào nhận xét về việc Ngữ cảnh thiếu gì.
- Nếu Ngữ cảnh thiếu chi tiết, vẫn trả lời đầy đủ bằng kiến thức pháp luật
  Việt Nam phổ thông, viết như một chuyên gia đang tư vấn.
- Cấm dùng dấu ngoặc vuông kiểu [1], [2] để trích dẫn.
- Cấm ký hiệu markdown ** và #.
- Cấm nhắc tới "Ngữ cảnh", "Văn bản 1", "Document ID" trong câu trả lời.
- Cấm mở đầu bằng "Dựa trên ngữ cảnh", "Câu trả lời là", hay bất kỳ câu nào
  nhận xét về chính câu trả lời.
- Không chép lại câu hỏi, không viết suy luận từng bước.
"""

# ── Safety ─────────────────────────────────────────────────────────────────────
ENV_SHA = "7ef96c552818e7ef87068a382cd75f25efe52a242ef293cae644cd1723104443"
CURRENT_VAL = "v1"
FROZEN_SHA = "8ed73d34e075b72557dd90970f6b486cefe554dba0fd55d6751b0537cfd92983"


def safety_check(label: str) -> None:
    p_env = ROOT / ".env"
    p_cur = ROOT / "storage/indexes/CURRENT"
    p_frz = ROOT / "data/evaluation/frozen_best.json"

    h_env = hashlib.sha256(p_env.read_bytes()).hexdigest() if p_env.is_file() else ""
    v_cur = p_cur.read_text(encoding="utf-8").strip() if p_cur.is_file() else ""
    h_frz = hashlib.sha256(p_frz.read_bytes()).hexdigest() if p_frz.is_file() else ""

    ok = h_env == ENV_SHA and v_cur == CURRENT_VAL and h_frz == FROZEN_SHA
    print(f"Safety [{label}]: .env={'OK' if h_env==ENV_SHA else 'FAIL'}  "
          f"CURRENT={'OK' if v_cur==CURRENT_VAL else 'FAIL'}  "
          f"frozen_best={'OK' if h_frz==FROZEN_SHA else 'FAIL'}")
    if not ok:
        print("ABORT — production state changed!", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 0: safety ───────────────────────────────────────────────────────
    safety_check("pre-run")

    # ── Phase 1: patch system prompt in-memory ────────────────────────────────
    import app.generation.prompts.system as prompt_mod
    prompt_mod.LEGAL_SYSTEM_PROMPT = V6_SYSTEM_PROMPT
    # Also patch the import cache in the prompt builder module
    import app.generation.prompts.legal_answer as builder_mod
    builder_mod.LEGAL_SYSTEM_PROMPT = V6_SYSTEM_PROMPT
    print("Patched system prompt in-memory.")

    # ── Phase 2: build runtime with widened retrieval ─────────────────────────
    from app.core.config import get_settings
    from app.services.runtime_factory import build_local_rag_runtime

    settings = get_settings()
    settings.index_root_dir = Path("storage/index-staging")
    settings.llm_adapter_path = "models/qlora-lr5e4/checkpoint-350"
    settings.min_new_tokens = 500

    runtime = build_local_rag_runtime(settings)

    # In case settings didn't propagate, force-set on the pipeline instance
    pipeline = runtime.service.retrieval_pipeline
    pipeline.dense_top_k = 50
    pipeline.bm25_top_k = 50
    pipeline.rrf_top_k = 100
    pipeline.reranker_top_k = 40
    print(f"Pipeline top-k: dense={pipeline.dense_top_k} bm25={pipeline.bm25_top_k} "
          f"rrf={pipeline.rrf_top_k} reranker={pipeline.reranker_top_k}")

    # ── Phase 3: write config JSON ────────────────────────────────────────────
    config = {
        "experiment": "V6",
        "checkpoint": "models/qlora-lr5e4/checkpoint-350",
        "LoRA": "r=8, alpha=16, dropout=0.05",
        "index": "storage/index-staging/v1-enriched",
        "dense_top_k": 50,
        "bm25_top_k": 50,
        "rrf_top_k": 100,
        "reranker_top_k": 40,
        "floor": 500,
        "prompt": "V6_INTERNAL_EVIDENCE_IDENTIFICATION",
        "context_budget": 10000,
        "neighbor_window": 1,
    }
    V6_CONFIG_JSON.write_text(json.dumps(config, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"Config written to {V6_CONFIG_JSON}")

    # ── Phase 4: load references ──────────────────────────────────────────────
    train = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    from app.submission.question_loader import QuestionDatasetLoader
    dataset = QuestionDatasetLoader().load(QUESTIONS_JSON)
    print(f"Loaded {len(dataset)} questions.")

    # ── Phase 5: run end-to-end generation ────────────────────────────────────
    submission: dict[str, dict] = {}
    per_q: list[dict] = []
    t_total = time.perf_counter()

    for i, query in enumerate(dataset, 1):
        t0 = time.perf_counter()
        answer = runtime.service.answer(query)
        dt = time.perf_counter() - t0

        qid = query.question_id
        pred = answer.answer
        submission[qid] = {"answer": pred}

        # article-hit check (cheap, no extra retrieval call)
        ref = train[qid]["answer"]
        gold_arts = set(ARTICLE_RE.findall(ref))
        pred_arts = set(ARTICLE_RE.findall(pred))
        art_hit = bool(gold_arts & pred_arts) or (not gold_arts and not pred_arts)

        per_q.append({
            "question_id": qid,
            "prediction": pred,
            "gold_articles": sorted(gold_arts),
            "pred_articles": sorted(pred_arts),
            "article_hit": art_hit,
            "latency_s": round(dt, 1),
        })
        print(f"[{i}/200] {qid}  {dt:.1f}s  art={'HIT' if art_hit else 'MISS'}")

    wall = time.perf_counter() - t_total
    print(f"\nDone in {wall:.0f}s  (avg {wall/len(dataset):.1f}s/q)")

    # ── Phase 6: write outputs ────────────────────────────────────────────────
    sub_path = OUT_DIR / "submission.json"
    sub_path.write_text(json.dumps(submission, ensure_ascii=False, indent=4),
                        encoding="utf-8")
    with V6_PER_Q_JSONL.open("w", encoding="utf-8") as fh:
        for row in per_q:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Submission → {sub_path}\nPer-question → {V6_PER_Q_JSONL}")

    # ── Phase 7: score ────────────────────────────────────────────────────────
    eval_cmd = [
        str(ROOT / ".venv/bin/python"),
        "scripts/eval_dev.py",
        "--score", str(sub_path),
        "--train", str(TRAIN_JSON),
        "--questions", str(QUESTIONS_JSON),
        "--allow-failures",
    ]
    ev = subprocess.run(eval_cmd, cwd=ROOT, capture_output=True, text=True)
    print(ev.stdout)
    if ev.stderr:
        print(ev.stderr, file=sys.stderr)

    m_meteor = re.search(r"METEOR\s+([\d.]+)", ev.stdout)
    m_rouge = re.search(r"ROUGE-L\s+([\d.]+)", ev.stdout)
    m_med = re.search(r"predicted median\s+(\d+)", ev.stdout)

    meteor = float(m_meteor.group(1)) if m_meteor else 0.0
    rouge = float(m_rouge.group(1)) if m_rouge else 0.0
    med_words = int(m_med.group(1)) if m_med else 0

    delta = meteor - 0.4880
    gate = delta >= 0.028
    verdict = "CANDIDATE WINNER" if gate else "NO-GO"

    art_hits = sum(1 for r in per_q if r["article_hit"])
    art_rate = art_hits / len(per_q)

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"METEOR  {meteor:.4f}  (baseline 0.4880  Δ {delta:+.4f})")
    print(f"ROUGE-L {rouge:.4f}  (baseline 0.3345)")
    print(f"Article Hit Rate: {art_rate:.4f} ({art_hits}/{len(per_q)})")
    print(f"{'='*60}")

    # ── Phase 8: build report ─────────────────────────────────────────────────
    lines = [
        "# V6 End-to-End Experiment Report", "",
        "## 1. Configuration", "",
        "| Parameter | Frozen v5 | V6 |",
        "|---|---|---|",
        "| dense_top_k | 20 | 50 |",
        "| bm25_top_k | 20 | 50 |",
        "| rrf_top_k | 30 | 100 |",
        "| reranker_top_k | 20 | 40 |",
        "| System Prompt | baseline | +internal evidence identification |",
        "| Everything else | identical | identical |", "",
        "## 2. Metrics", "",
        "| Metric | Frozen v5 | V6 | Delta |",
        "|---|---|---|---|",
        f"| **METEOR** | 0.4880 | {meteor:.4f} | {delta:+.4f} |",
        f"| **ROUGE-L** | 0.3345 | {rouge:.4f} | {rouge-0.3345:+.4f} |",
        f"| Median Words | 457 | {med_words} | {med_words-457:+} |",
        f"| Article Hit Rate | — | {art_rate:.4f} ({art_hits}/200) | — |", "",
        "## 3. Decision Gate", "",
        f"- Gate threshold: +0.028 METEOR",
        f"- Actual delta:   {delta:+.4f}",
        f"- **Verdict: `{verdict}`**", "",
        "## 4. Runtime", "",
        f"- Wall time: {wall:.0f}s ({wall/60:.1f}min)",
        f"- Avg per query: {wall/len(dataset):.1f}s", "",
    ]

    # ── Phase 9: safety post-check ────────────────────────────────────────────
    safety_check("post-run")
    lines += [
        "## 5. Safety", "",
        f"- .env SHA prefix: `{ENV_SHA[:16]}` ✓",
        f"- CURRENT: `{CURRENT_VAL}` ✓",
        f"- frozen_best SHA prefix: `{FROZEN_SHA[:16]}` ✓", "",
    ]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report → {REPORT_MD}")


if __name__ == "__main__":
    main()
