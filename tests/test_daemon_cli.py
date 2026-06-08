# tests/test_daemon_cli.py
import sys

def test_daemon_subcommand_invokes_run_daemon(monkeypatch):
    import brain_mcp.daemon.server as srv
    called = {}
    monkeypatch.setattr(srv, "run_daemon", lambda: called.__setitem__("ran", True))
    monkeypatch.setattr(sys, "argv", ["brain_mcp", "daemon"])
    from brain_mcp.__main__ import main
    main()
    assert called.get("ran") is True

def test_daemon_default_invocation(monkeypatch):
    import brain_mcp.daemon.server as srv
    called = {}
    monkeypatch.setattr(srv, "run_daemon", lambda: called.__setitem__("ran", True))
    monkeypatch.setattr(sys, "argv", ["brain_mcp", "daemon"])
    from brain_mcp.__main__ import main
    main()
    assert called.get("ran") is True
