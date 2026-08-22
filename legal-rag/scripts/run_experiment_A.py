import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "data/outputs/experiment_A"
QUESTIONS_PATH = ROOT / "data/questions/dev200.json"
TRAIN_PATH = ROOT / "data/train/train.json"
REPORT_PATH = ROOT / "data/evaluation/experiment_A_article_selection.md"

BASELINE_METEOR = 0.4880
BASELINE_ROUGE_L = 0.3345

ARTICLE_RE = re.compile(r"Điều\s+(\d+)")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup environment variables for baseline run
    env = os.environ.copy()
    env["INDEX_ROOT_DIR"] = "storage/index-staging"
    env["RERANKER_TOP_K"] = "20"
    env["LLM_BATCH_SIZE"] = "1"
    env["LLM_ADAPTER_PATH"] = "models/qlora-lr5e4/checkpoint-350"
    env["LLM_MIN_NEW_TOKENS"] = "500"
    env["CONTEXT_NEIGHBOR_WINDOW"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    
    # 2. Run the submission script on the 200 questions
    cmd = [
        str(ROOT / ".venv/bin/python"),
        "scripts/run_ingestion.py",
        "submit",
        "--questions", str(QUESTIONS_PATH),
        "--output", str(OUT_DIR / "submission.json"),
        "--checkpoint", str(OUT_DIR / "partial.jsonl")
    ]
    
    print("Running Experiment A generation on 200 questions...")
    result = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print("Experiment execution failed!", file=sys.stderr)
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
        
    print("Experiment generation completed. Scoring outputs...")

    # 3. Score using eval_dev.py
    eval_cmd = [
        str(ROOT / ".venv/bin/python"),
        "scripts/eval_dev.py",
        "--score", str(OUT_DIR / "submission.json"),
        "--train", str(TRAIN_PATH),
        "--questions", str(QUESTIONS_PATH)
    ]
    eval_res = subprocess.run(eval_cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    print(eval_res.stdout)
    
    # Parse scores
    meteor_match = re.search(r"METEOR\s+([\d.]+)", eval_res.stdout)
    rouge_match = re.search(r"ROUGE-L\s+([\d.]+)", eval_res.stdout)
    median_match = re.search(r"predicted median\s+(\d+)", eval_res.stdout)
    
    meteor_score = float(meteor_match.group(1)) if meteor_match else 0.0
    rouge_score = float(rouge_match.group(1)) if rouge_match else 0.0
    median_words = int(median_match.group(1)) if median_match else 0
    
    # 4. Load train references and calculate article hit rate
    train = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    submission = json.loads((OUT_DIR / "submission.json").read_text(encoding="utf-8"))
    
    total_questions = len(submission)
    hits = 0
    detailed_hits = []
    
    for qid, record in submission.items():
        if qid not in train:
            continue
        model_answer = record["answer"]
        reference_answer = train[qid]["answer"]
        
        # Extract gold articles: e.g. "Điều 5"
        gold_articles = set(ARTICLE_RE.findall(reference_answer))
        # Extract model cited articles: e.g. "Điều 5"
        model_articles = set(ARTICLE_RE.findall(model_answer))
        
        hit = len(gold_articles & model_articles) > 0 or (not gold_articles and not model_articles)
        if hit:
            hits += 1
            
        detailed_hits.append({
            "question_id": qid,
            "gold_articles": sorted(gold_articles),
            "model_articles": sorted(model_articles),
            "hit": hit
        })
        
    hit_rate = hits / total_questions if total_questions > 0 else 0.0
    print(f"Article Selection Hit Rate: {hit_rate:.4f} ({hits}/{total_questions})")
    
    # 5. Check Gate
    diff_meteor = meteor_score - BASELINE_METEOR
    gate_passed = diff_meteor >= 0.028
    verdict = "GO (Promising)" if gate_passed else "NO-GO (Not enough improvement)"
    
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"METEOR: {meteor_score:.4f} (Baseline: {BASELINE_METEOR:.4f}, Delta: {diff_meteor:+.4f})")
    print(f"ROUGE-L: {rouge_score:.4f} (Baseline: {BASELINE_ROUGE_L:.4f})")
    
    # 6. Generate Report
    report_lines = [
        "# Experiment A — Article-Selection Prompt Ablation",
        "",
        "## 1. Parameters & Configuration",
        f"- **Checkpoint**: `models/qlora-lr5e4/checkpoint-350`",
        f"- **LoRA**: r=8, alpha=16, dropout=0.05",
        f"- **Index**: `storage/index-staging/v1-enriched`",
        f"- **k**: `20`",
        f"- **LLM_MIN_NEW_TOKENS**: `500`",
        f"- **Context window**: `10000` tokens",
        f"- **CONTEXT_NEIGHBOR_WINDOW**: `1`",
        "",
        "## 2. Quantitative Results",
        "| Metric | Baseline (Frozen Winner) | Experiment A (New Prompt) | Delta |",
        "|---|---|---|---|",
        f"| **METEOR** | {BASELINE_METEOR:.4f} | {meteor_score:.4f} | {diff_meteor:+.4f} |",
        f"| **ROUGE-L** | {BASELINE_ROUGE_L:.4f} | {rouge_score:.4f} | {rouge_score - BASELINE_ROUGE_L:+.4f} |",
        f"| **Median Words** | 457 | {median_words} | {median_words - 457:+} |",
        f"| **Article Hit Rate** | N/A | {hit_rate:.4f} ({hits}/{total_questions}) | - |",
        "",
        "## 3. Verdict",
        f"**Decision Gate Verdict**: `{verdict}`",
        f"- Improvement limit (1 SE of dev200): `+0.028`",
        f"- Actual METEOR Delta: `{diff_meteor:+.4f}`",
        "",
        "## 4. Diagnostics per Question",
        "| Question ID | Gold Articles | Model Articles | Hit |",
        "|---|---|---|---|",
    ]
    for row in detailed_hits:
        report_lines.append(
            f"| {row['question_id']} | {', '.join(row['gold_articles'])} | "
            f"{', '.join(row['model_articles'])} | {'PASS' if row['hit'] else 'FAIL'} |"
        )
        
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    
    # Save the hit diagnostics to the outputs folder
    with open(OUT_DIR / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(detailed_hits, f, ensure_ascii=False, indent=2)
        
    # Write a summary result JSON for easier validation
    summary = {
        "meteor": meteor_score,
        "rouge_l": rouge_score,
        "median_words": median_words,
        "article_hit_rate": hit_rate,
        "meteor_delta": diff_meteor,
        "gate_passed": gate_passed,
        "verdict": verdict
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
