"""Download files from the rented box over SFTP.

The mirror of scripts/remote_put.py; credentials come from the same gitignored
data/outputs/remote.env.

Usage:
  python scripts/remote_get.py <remote_path> <local_dir>
"""

import argparse
import os
import time
from pathlib import Path

from remote import load_credentials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_path")
    parser.add_argument("local_dir", type=Path)
    args = parser.parse_args()

    load_credentials()
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ["REMOTE_HOST"],
        port=int(os.environ["REMOTE_PORT"]),
        username=os.environ["REMOTE_USER"],
        password=os.environ["REMOTE_PASSWORD"],
        timeout=30,
    )
    sftp = client.open_sftp()
    try:
        size = sftp.stat(args.remote_path).st_size
        args.local_dir.mkdir(parents=True, exist_ok=True)
        target = args.local_dir / Path(args.remote_path).name
        started = time.perf_counter()
        sftp.get(args.remote_path, str(target))
        elapsed = time.perf_counter() - started
        print(
            f"{target} | {size / 1e6:.1f} MB in {elapsed:.0f}s "
            f"-> {size / 1e6 / max(elapsed, 1e-9):.1f} MB/s"
        )
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
