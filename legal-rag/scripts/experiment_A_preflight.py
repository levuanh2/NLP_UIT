import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREFLIGHT_DIR = ROOT / "data/outputs/experiment_A_preflight"
BASELINE_PATH = ROOT / "data/outputs/preflight-dev200-enriched-k20-ckpt350/partial.jsonl"
QUESTIONS_PATH = ROOT / "data/questions/dev5.json"

def main():
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup environment variables for baseline run
    env = os.environ.copy()
    env["INDEX_ROOT_DIR"] = "storage/index-staging"
    env["RERANKER_TOP_K"] = "20"
    env["LLM_BATCH_SIZE"] = "1"
    env["LLM_ADAPTER_PATH"] = "models/qlora-lr5e4/checkpoint-350"
    env["LLM_MIN_NEW_TOKENS"] = "500"
    env["CONTEXT_NEIGHBOR_WINDOW"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    
    # 2. Run the submission script on the 5 preflight questions
    cmd = [
        str(ROOT / ".venv/bin/python"),
        "scripts/run_ingestion.py",
        "submit",
        "--questions", str(QUESTIONS_PATH),
        "--output", str(PREFLIGHT_DIR / "submission.json"),
        "--checkpoint", str(PREFLIGHT_DIR / "partial.jsonl")
    ]
    
    print("Running preflight generation on 5 questions...")
    result = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print("Preflight execution failed!", file=sys.stderr)
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
        
    print("Preflight execution succeeded. Loading outputs for comparison...")
    
    # 3. Load baseline partial.jsonl
    baseline_evidences = {}
    if not BASELINE_PATH.is_file():
        print(f"ERROR: Baseline file not found: {BASELINE_PATH}", file=sys.stderr)
        sys.exit(2)
        
    for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        baseline_evidences[row["question_id"]] = row.get("evidence_ids", [])
        
    # 4. Load preflight partial.jsonl
    preflight_path = PREFLIGHT_DIR / "partial.jsonl"
    preflight_evidences = {}
    for line in preflight_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        preflight_evidences[row["question_id"]] = row.get("evidence_ids", [])
        
    # 5. Compare evidence_ids
    print("\n=== RETRIEVAL COMPARISON ===")
    mismatch = False
    for qid in preflight_evidences:
        base_ev = baseline_evidences.get(qid, [])
        pref_ev = preflight_evidences.get(qid, [])
        print(f"Question {qid}:")
        print(f"  Baseline:  {base_ev}")
        print(f"  Preflight: {pref_ev}")
        if base_ev != pref_ev:
            print("  ==> MISMATCH! Retrieval outputs differ.", file=sys.stderr)
            mismatch = True
        else:
            print("  ==> MATCH.")
            
    if mismatch:
        print("\n[PREFLIGHT FAILED] Ranking/retrieval output is NOT identical to baseline.", file=sys.stderr)
        sys.exit(3)
    else:
        print("\n[PREFLIGHT SUCCESSFUL] Ranking/retrieval output is 100% identical to baseline.")
        sys.exit(0)

if __name__ == "__main__":
    main()
