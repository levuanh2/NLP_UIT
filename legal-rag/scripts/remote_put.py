"""Upload files to the rented box over SFTP, with resume and progress.

Credentials come from data/outputs/remote.env (gitignored); see scripts/remote.py.

Usage:
  python scripts/remote_put.py <local_path> <remote_dir> [--limit-bytes N]
"""

import argparse
import os
import stat
import time
from pathlib import Path

from remote import load_credentials


def connect():
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ["REMOTE_HOST"],
        port=int(os.environ["REMOTE_PORT"]),
        username=os.environ["REMOTE_USER"],
        password=os.environ["REMOTE_PASSWORD"],
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def remote_size(sftp, path: str) -> int:
    try:
        return sftp.stat(path).st_size
    except OSError:
        return -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("local", type=Path)
    parser.add_argument("remote_dir")
    parser.add_argument(
        "--limit-bytes",
        type=int,
        default=0,
        help="Stop after this many bytes; used to measure throughput.",
    )
    args = parser.parse_args()

    load_credentials()
    files = (
        sorted(p for p in args.local.rglob("*") if p.is_file())
        if args.local.is_dir()
        else [args.local]
    )
    base = args.local if args.local.is_dir() else args.local.parent
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} file(s), {total / 2**30:.2f} GiB")

    client = connect()
    sftp = client.open_sftp()
    sent = 0
    started = time.perf_counter()
    try:
        for path in files:
            relative = path.relative_to(base).as_posix()
            target = f"{args.remote_dir}/{relative}"
            parent = target.rsplit("/", 1)[0]
            # mkdir -p, since SFTP has no recursive create
            built = ""
            for part in parent.split("/"):
                built = f"{built}/{part}" if built else part
                try:
                    sftp.stat(built)
                except OSError:
                    sftp.mkdir(built)
            size = path.stat().st_size
            if remote_size(sftp, target) == size:
                print(f"  skip {relative}")
                continue
            file_started = time.perf_counter()
            sftp.put(str(path), target)
            elapsed = time.perf_counter() - file_started
            sent += size
            rate = size / elapsed / 2**20 if elapsed else 0
            print(f"  {relative}  {size / 2**20:.0f} MiB  {rate:.1f} MiB/s")
            if args.limit_bytes and sent >= args.limit_bytes:
                print("  (limit reached)")
                break
        mode = sftp.stat(args.remote_dir).st_mode
        assert stat.S_ISDIR(mode), args.remote_dir
    finally:
        sftp.close()
        client.close()

    elapsed = time.perf_counter() - started
    rate = sent / elapsed / 2**20 if elapsed else 0
    print(f"sent {sent / 2**20:.0f} MiB in {elapsed:.0f}s -> {rate:.1f} MiB/s")
    if rate and total > sent:
        print(f"remaining {(total - sent) / 2**20:.0f} MiB "
              f"-> ~{(total - sent) / (rate * 2**20) / 60:.0f} min at this rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
