# tests/test_daemon_registry.py
from brain_mcp.daemon.registry import write_registry, read_registry, DaemonInfo

def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / "daemon.json"
    write_registry(p, DaemonInfo(port=47611, token="secrettok", pid=1234))
    info = read_registry(p)
    assert info is not None
    assert info.port == 47611 and info.token == "secrettok" and info.pid == 1234

def test_read_missing_returns_none(tmp_path):
    assert read_registry(tmp_path / "nope.json") is None

def test_read_corrupt_returns_none(tmp_path):
    p = tmp_path / "daemon.json"
    p.write_text("not json{", encoding="utf-8")
    assert read_registry(p) is None

def test_file_is_owner_only_readable(tmp_path):
    import os, stat
    p = tmp_path / "daemon.json"
    write_registry(p, DaemonInfo(port=1, token="t", pid=2))
    mode = stat.S_IMODE(os.stat(p).st_mode)
    # On Windows os.chmod is effectively a no-op; only assert 0o600 on POSIX.
    assert os.name == "nt" or mode == 0o600
