# tests/test_daemon_cli.py
import sys


def test_daemon_subcommand_passes_idle(monkeypatch):
    import brain_mcp.daemon.server as srv
    got = {}
    monkeypatch.setattr(srv, "run_daemon", lambda idle_timeout=1800.0: got.__setitem__("idle", idle_timeout))
    monkeypatch.setattr(sys, "argv", ["brain_mcp", "daemon", "--idle", "5"])
    from brain_mcp.__main__ import main
    main()
    assert got["idle"] == 5.0


def test_daemon_default_idle(monkeypatch):
    import brain_mcp.daemon.server as srv
    got = {}
    monkeypatch.setattr(srv, "run_daemon", lambda idle_timeout=1800.0: got.__setitem__("idle", idle_timeout))
    monkeypatch.setattr(sys, "argv", ["brain_mcp", "daemon"])
    from brain_mcp.__main__ import main
    main()
    assert got["idle"] == 1800.0
