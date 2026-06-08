import json, os, signal, subprocess, sys, time
import pytest
from pathlib import Path

@pytest.mark.slow
def test_proxy_end_to_end(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    vault = tmp_path / "vault"; vault.mkdir()
    (vault / "n.md").write_text("# hi\nnote about routing\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"vault_path": str(vault), "auto_index": False, "index_on_startup": False}),
        encoding="utf-8")
    env = dict(os.environ, BRAIN_DATA_DIR=str(tmp_path), BRAIN_VAULT_PATH=str(vault),
               PYTHONPATH=str(repo), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proxy = subprocess.Popen([sys.executable, "-m", "brain_mcp", "serve"], cwd=str(repo), env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, bufsize=1)
    try:
        msgs = [
            {"jsonrpc":"2.0","id":1,"method":"initialize",
             "params":{"protocolVersion":"2025-06-18","capabilities":{},
                       "clientInfo":{"name":"t","version":"1"}}},
            {"jsonrpc":"2.0","method":"notifications/initialized"},
            {"jsonrpc":"2.0","id":2,"method":"tools/list"},
        ]
        for m in msgs:
            proxy.stdin.write(json.dumps(m) + "\n"); proxy.stdin.flush()
        # Read responses for id 1 and id 2 (notifications produce no output).
        seen = {}
        deadline = time.time() + 40
        while time.time() < deadline and {1,2} - set(seen):
            line = proxy.stdout.readline()
            if not line:
                break
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue
            if "id" in obj: seen[obj["id"]] = obj
        assert 1 in seen and "result" in seen[1], f"no initialize result: {seen}"
        assert 2 in seen and "result" in seen[2], f"no tools/list result: {seen}"
        names = {t["name"] for t in seen[2]["result"]["tools"]}
        assert "brain_retrieve" in names and "brain_store" in names, names
    finally:
        try: proxy.stdin.close()
        except Exception: pass
        proxy.terminate()
        try: proxy.wait(timeout=10)
        except Exception: proxy.kill()
        # kill the daemon the proxy started
        try:
            d = json.loads((tmp_path / "daemon.json").read_text(encoding="utf-8"))
            os.kill(int(d["pid"]), signal.SIGTERM)
        except Exception:
            pass


@pytest.mark.slow
def test_proxy_replies_even_if_daemon_dies_midsession(tmp_path):
    import json, os, signal, subprocess, sys, time
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    vault = tmp_path / "vault"; vault.mkdir()
    (vault / "n.md").write_text("# hi\nnote\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"vault_path": str(vault), "auto_index": False, "index_on_startup": False}),
        encoding="utf-8")
    env = dict(os.environ, BRAIN_DATA_DIR=str(tmp_path), BRAIN_VAULT_PATH=str(vault),
               PYTHONPATH=str(repo), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    p = subprocess.Popen([sys.executable, "-m", "brain_mcp", "serve"], cwd=str(repo), env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1)
    def send(m): p.stdin.write(json.dumps(m) + "\n"); p.stdin.flush()
    def read_id(target, timeout=40):
        end = time.time() + timeout
        while time.time() < end:
            line = p.stdout.readline()
            if not line: return None
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue
            if obj.get("id") == target: return obj
        return None
    pid1 = None
    try:
        send({"jsonrpc":"2.0","id":1,"method":"initialize",
              "params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}})
        assert read_id(1) is not None, "initialize failed"
        d = json.loads((tmp_path / "daemon.json").read_text(encoding="utf-8")); pid1 = int(d["pid"])
        os.kill(pid1, signal.SIGTERM)          # daemon dies mid-session
        time.sleep(1.0)
        send({"jsonrpc":"2.0","method":"notifications/initialized"})
        send({"jsonrpc":"2.0","id":2,"method":"tools/list"})
        resp = read_id(2)                       # MUST get a reply (respawned result OR error), not hang
        assert resp is not None and ("result" in resp or "error" in resp), f"client hung / bad reply: {resp}"
    finally:
        try: p.stdin.close()
        except Exception: pass
        p.terminate()
        try: p.wait(timeout=10)
        except Exception: p.kill()
        for pidfile in (tmp_path / "daemon.json",):
            try:
                d = json.loads(pidfile.read_text(encoding="utf-8")); os.kill(int(d["pid"]), signal.SIGTERM)
            except Exception: pass
