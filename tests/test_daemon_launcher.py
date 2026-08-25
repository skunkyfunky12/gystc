import os
import signal
import pytest
from brain_mcp.daemon.registry import DaemonInfo
from brain_mcp.daemon.launcher import daemon_health_url

def test_health_url_from_info():
    assert daemon_health_url(DaemonInfo(port=47611, token="t", pid=1)) == "http://127.0.0.1:47611/health"

@pytest.mark.slow
def test_ensure_daemon_spawns_and_reuses(tmp_path, monkeypatch):
    # Isolate: temp data dir + tiny vault; no indexing so startup is fast.
    vault = tmp_path / "vault"; vault.mkdir()
    (vault / "n.md").write_text("# hi\nnote\n", encoding="utf-8")
    import json
    (tmp_path / "config.json").write_text(
        json.dumps({"vault_path": str(vault), "auto_index": False, "index_on_startup": False}),
        encoding="utf-8")
    monkeypatch.setenv("BRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAIN_VAULT_PATH", str(vault))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from brain_mcp.daemon.launcher import ensure_daemon, is_alive
    a = None
    try:
        a = ensure_daemon(tmp_path)
        assert a is not None and is_alive(a), "daemon did not come up"
        b = ensure_daemon(tmp_path)
        assert b is not None and b.pid == a.pid, "second call must reuse the same daemon"
    finally:
        if a is not None:
            try: os.kill(a.pid, signal.SIGTERM)
            except OSError: pass
