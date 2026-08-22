"""Run a command on the rented GPU box over SSH.

Credentials live in data/outputs/remote.env (gitignored) so they stay out of
the repo and out of shell history:

    REMOTE_HOST=...
    REMOTE_PORT=...
    REMOTE_USER=...
    REMOTE_PASSWORD=...

Usage: python scripts/remote.py "<command>" [--timeout SECONDS]
"""

import argparse
import os
import sys
from pathlib import Path

CREDENTIALS = Path(__file__).resolve().parents[1] / "data/outputs/remote.env"


def load_credentials() -> None:
    """Fill REMOTE_* from the credentials file without overriding the env."""
    if not CREDENTIALS.is_file():
        return
    for line in CREDENTIALS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    load_credentials()
    missing = [
        name
        for name in ("REMOTE_HOST", "REMOTE_PORT", "REMOTE_USER", "REMOTE_PASSWORD")
        if not os.environ.get(name)
    ]
    if missing:
        print(f"missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2

    import paramiko

    client = paramiko.SSHClient()
    # ponytail: a rented box has no known_hosts entry yet; pin it once the
    # fingerprint matters more than getting connected.
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
    try:
        _, stdout, stderr = client.exec_command(args.command, timeout=args.timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        status = stdout.channel.recv_exit_status()
    finally:
        client.close()

    if out:
        print(out.rstrip())
    if err:
        print(err.rstrip(), file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
