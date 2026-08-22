import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ENV_SHA = "7ef96c552818e7ef87068a382cd75f25efe52a242ef293cae644cd1723104443"
EXPECTED_CUR_SHA = "2d27fbdf4e8ca207afbfa388ca9172fbcc6c70e534af2476b3b704f87debadcf"
EXPECTED_CKPT_SIZE = 59933632
EXPECTED_INDEX_SIZE = 1204154368

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    errors = []
    print("=== ENVIRONMENT VERIFICATION ===")
    
    # 1. Verify .env SHA
    env_path = ROOT / ".env"
    if not env_path.is_file():
        errors.append(".env file is missing!")
    else:
        env_sha = sha256(env_path)
        print(f".env SHA-256: {env_sha}")
        if env_sha != EXPECTED_ENV_SHA:
            errors.append(f".env SHA mismatch! Found {env_sha[:16]}, expected {EXPECTED_ENV_SHA[:16]}")
            
    # 2. Verify CURRENT SHA
    cur_path = ROOT / "storage/indexes/CURRENT"
    if not cur_path.is_file():
        errors.append("storage/indexes/CURRENT is missing!")
    else:
        cur_sha = sha256(cur_path)
        print(f"indexes/CURRENT SHA-256: {cur_sha}")
        if cur_sha != EXPECTED_CUR_SHA:
            errors.append(f"CURRENT SHA mismatch! Found {cur_sha[:16]}, expected {EXPECTED_CUR_SHA[:16]}")
            
    # 3. Verify Checkpoint Fingerprint
    ckpt_file = ROOT / "models/qlora-lr5e4/checkpoint-350/adapter_model.safetensors"
    if not ckpt_file.is_file():
        errors.append("Checkpoint adapter_model.safetensors is missing!")
    else:
        size = ckpt_file.stat().st_size
        print(f"Checkpoint adapter size: {size} bytes")
        if size != EXPECTED_CKPT_SIZE:
            errors.append(f"Checkpoint adapter size mismatch! Found {size}, expected {EXPECTED_CKPT_SIZE}")

    # 4. Verify Index Fingerprint
    index_file = ROOT / "storage/index-staging/v1-enriched/bm25/bm25.sqlite"
    if not index_file.is_file():
        errors.append("Staging index bm25.sqlite is missing!")
    else:
        size = index_file.stat().st_size
        print(f"Staging index size: {size} bytes")
        if size != EXPECTED_INDEX_SIZE:
            errors.append(f"Staging index size mismatch! Found {size}, expected {EXPECTED_INDEX_SIZE}")

    # 5. Verify Git Diff
    try:
        git_diff = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True
        )
        modified = [line.strip() for line in git_diff.stdout.splitlines() if line.strip()]
        print("Modified files:")
        for f in modified:
            print(f"  - {f}")
    except Exception as e:
        errors.append(f"Failed to check git diff: {e}")

    if errors:
        print("\n[VERIFICATION FAILED]")
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n[VERIFICATION SUCCESSFUL] All hashes, fingerprints and configurations match baseline.")
        sys.exit(0)

if __name__ == "__main__":
    main()
