from __future__ import annotations
import json, sys
import httpx
from brain_mcp.config import load_config
from brain_mcp.daemon.launcher import ensure_daemon

_HEADERS = {"Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"}


def forward_one(url: str, token: str, msg: dict, client: httpx.Client | None = None) -> dict | None:
    """POST one JSON-RPC message to the daemon; return the JSON response, or None
    for notifications / 202 (no response expected)."""
    own = client is None
    client = client or httpx.Client(timeout=60.0)
    try:
        r = client.post(url, headers={**_HEADERS, "Authorization": f"Bearer {token}"}, json=msg)
        if r.status_code == 202 or not r.content:
            return None
        return r.json()
    finally:
        if own:
            client.close()


def run_proxy() -> None:
    data_dir = load_config().data_dir
    info = ensure_daemon(data_dir)
    if info is None:
        print("FATAL: could not reach or start the GYSTC daemon", file=sys.stderr)
        sys.exit(1)
    url = f"http://127.0.0.1:{info.port}/mcp"
    out = sys.stdout
    # IMPORTANT: use readline(), NOT `for line in sys.stdin` (the latter buffers
    # and would stall interactive stdio).
    with httpx.Client(timeout=60.0) as client:
        while True:
            line = sys.stdin.readline()
            if not line:
                break  # EOF: Claude Code closed the pipe
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                resp = forward_one(url, info.token, msg, client=client)
            except Exception as exc:
                print(f"proxy: forward failed: {exc}", file=sys.stderr)
                continue
            if resp is not None:
                out.write(json.dumps(resp) + "\n")
                out.flush()
