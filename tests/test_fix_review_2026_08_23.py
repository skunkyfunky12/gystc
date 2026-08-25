"""Regression tests for the remaining findings of the external review 2026-08-23.

Finding 5:  sanitize_title lets Windows device names through (CON, NUL, COM1...).
Finding 6:  the watcher misses `.MD`, and on_deleted checks neither excludes nor pending.
Finding 7:  created_at is the modification time, so the real creation time is lost.
Finding 8:  claimed "/health is unauthenticated" -- pinned here, see the test.
Finding 9:  forward_one passes an HTTP error body off as a JSON-RPC response.
Finding 11: load_config never validates what it loaded.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, UTC

import httpx
import pytest


# ------------------------------------------------------------------ finding 5

WINDOWS_DEVICE_NAMES = ["CON", "con", "PRN", "AUX", "NUL", "COM1", "lpt9", "CON.backup"]
RESERVED = {"CON", "PRN", "AUX", "NUL", "COM1", "LPT9"}


@pytest.mark.parametrize("title", WINDOWS_DEVICE_NAMES)
def test_sanitize_title_escapes_windows_device_names(title):
    """CON.md cannot be created on Windows -- the OS resolves the name to the
    console device. The store fails gracefully, but a title the user typed
    should still end up as a file."""
    from brain_mcp.tools.store import sanitize_title

    out = sanitize_title(title)

    stem = out.split(".")[0].upper()
    assert stem not in RESERVED, f"{title!r} became {out!r}, still a device name"
    assert out, "the title must not be emptied either"


def test_sanitize_title_leaves_ordinary_titles_alone():
    from brain_mcp.tools.store import sanitize_title

    assert sanitize_title("Concurrency notes") == "Concurrency notes"
    assert sanitize_title("COMET") == "COMET"


# ------------------------------------------------------------------ finding 6

@pytest.fixture
def watcher_handler(tmp_path):
    """A watcher handler over tmp_path that records what it dispatches."""
    from brain_mcp.indexer.watcher import _Handler

    seen: list[tuple[str, str]] = []
    handler = _Handler(
        on_change=lambda path, event: seen.append((path, event)),
        pending={}, lock=threading.Lock(), vault_root=tmp_path.resolve(),
        exclude_dirs=["99 Archiv"],
    )
    return handler, seen, tmp_path


@pytest.mark.parametrize("suffix", [".md", ".MD", ".Md"])
def test_watcher_accepts_markdown_in_any_case(watcher_handler, suffix):
    from watchdog.events import FileCreatedEvent
    handler, seen, tmp_path = watcher_handler
    path = str(tmp_path / ("note" + suffix))

    handler.on_created(FileCreatedEvent(path))

    assert seen == [(path, "created")], "suffix " + suffix + " was ignored"


def test_watcher_ignores_non_markdown(watcher_handler):
    from watchdog.events import FileCreatedEvent
    handler, seen, tmp_path = watcher_handler

    handler.on_created(FileCreatedEvent(str(tmp_path / "image.png")))

    assert seen == []


def test_watcher_delete_respects_excluded_dirs(watcher_handler):
    """Archiving a folder deletes many notes at once. Dispatching those pulls
    exactly the notes the exclude list exists to keep out back through the
    change path."""
    from watchdog.events import FileDeletedEvent
    handler, seen, tmp_path = watcher_handler
    archived = tmp_path / "99 Archiv" / "old.md"
    archived.parent.mkdir(parents=True)
    archived.write_text("x", encoding="utf-8")

    handler.on_deleted(FileDeletedEvent(str(archived)))

    assert seen == []


def test_watcher_delete_respects_pending_writes(tmp_path):
    """A path the store is mid-write on must not be dispatched as a delete --
    the same registry that already guards created and modified."""
    from watchdog.events import FileDeletedEvent
    from brain_mcp.indexer.watcher import _Handler

    seen: list[tuple[str, str]] = []
    target = tmp_path / "note.md"
    target.write_text("x", encoding="utf-8")
    handler = _Handler(
        on_change=lambda path, event: seen.append((path, event)),
        pending={str(target.resolve()): time.time()},
        lock=threading.Lock(), vault_root=tmp_path.resolve(),
    )

    handler.on_deleted(FileDeletedEvent(str(target)))

    assert seen == []


# ------------------------------------------------------------------ finding 7

def test_created_at_is_not_taken_from_the_modification_time(tmp_path):
    """A note edited today keeps the date it was created. Pinned by pushing the
    modification time into the future: with created_at == mtime the scanner
    reports a creation date that has not happened yet."""
    from brain_mcp.indexer.scanner import parse_note_file

    note = tmp_path / "note.md"
    note.write_text("# Title\nbody\n", encoding="utf-8")
    future = datetime.now(tz=UTC) + timedelta(days=400)
    os.utime(note, (future.timestamp(), future.timestamp()))

    data = parse_note_file(note, tmp_path, {})

    future_date = future.strftime("%Y-%m-%d")
    assert data["created_at"] < future_date, (
        "created_at " + data["created_at"] + " is the (future) modification time")
    assert data["modified_at"].startswith(future_date)


# ------------------------------------------------------------------ finding 8

def test_daemon_guard_covers_health_route_appended_after_the_middleware():
    """The review filed /health as unauthenticated (leaks the PID). Production
    appends the route to app.router.routes AFTER add_middleware, which is the
    ordering that could plausibly bypass the guard -- reproduced exactly here so
    the answer is measured rather than reasoned."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from brain_mcp.daemon.server import build_guard_middleware

    token = secrets.token_hex(16)
    app = Starlette()
    app.add_middleware(build_guard_middleware(token=token, allowed_origins=[]))

    async def health(_req):
        return JSONResponse({"ok": True, "pid": 424242})

    app.router.routes.append(Route("/health", health, methods=["GET"]))

    client = TestClient(app)
    anonymous = client.get("/health")
    assert anonymous.status_code == 401
    assert b"424242" not in anonymous.content, "the PID leaked to an unauthenticated caller"

    authorized = client.get("/health", headers={"Authorization": "Bearer " + token})
    assert authorized.status_code == 200
    assert authorized.json()["pid"] == 424242


# ------------------------------------------------------------------ finding 9

class _StubClient:
    """httpx.Client stand-in returning one canned response."""

    def __init__(self, response):
        self._response = response
        self.closed = False

    def post(self, url, headers=None, json=None):
        return self._response

    def close(self):
        self.closed = True


def _response(status, payload, url="http://127.0.0.1:1/mcp"):
    return httpx.Response(status, json=payload, request=httpx.Request("POST", url))


def test_forward_one_does_not_pass_an_http_error_off_as_a_response():
    """A 401 from a stale token used to be returned verbatim, so the client saw
    an error object where a JSON-RPC response belongs. Raising lets
    _forward_with_recovery respawn the daemon, which is what a stale token
    needs."""
    from brain_mcp.daemon.proxy import forward_one

    client = _StubClient(_response(401, {"error": "unauthorized"}))

    with pytest.raises(httpx.HTTPStatusError):
        forward_one("http://127.0.0.1:1/mcp", "tok", {"jsonrpc": "2.0", "id": 1},
                    client=client)


def test_forward_one_returns_a_successful_response():
    from brain_mcp.daemon.proxy import forward_one

    payload = {"jsonrpc": "2.0", "id": 1, "result": {}}
    client = _StubClient(_response(200, payload))

    assert forward_one("http://127.0.0.1:1/mcp", "tok", {"jsonrpc": "2.0", "id": 1},
                       client=client) == payload


def test_forward_one_returns_none_for_a_notification():
    from brain_mcp.daemon.proxy import forward_one

    client = _StubClient(httpx.Response(202, request=httpx.Request("POST", "http://x/mcp")))

    assert forward_one("http://x/mcp", "tok", {"jsonrpc": "2.0"}, client=client) is None


# ----------------------------------------------------------------- finding 11

def test_load_config_reports_an_invalid_stored_value(tmp_path, monkeypatch, capsys):
    """validate_config only ran in save_config, so a config.json edited by hand
    (or written by an older version) loaded silently and misbehaved later."""
    from brain_mcp.config import load_config

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.json").write_text(
        json.dumps({"log_level": "LOUD", "folder_to_region": {"docs": 99}}),
        encoding="utf-8")
    monkeypatch.setenv("BRAIN_DATA_DIR", str(data_dir))
    monkeypatch.delenv("BRAIN_LOG_LEVEL", raising=False)

    config = load_config()

    err = capsys.readouterr().err
    assert "log_level" in err and "LOUD" in err
    assert "folder_to_region" in err
    assert config.data_dir == data_dir, "a warning must not stop the config from loading"


def test_load_config_stays_quiet_for_a_valid_file(tmp_path, monkeypatch, capsys):
    from brain_mcp.config import load_config

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.json").write_text(
        json.dumps({"log_level": "DEBUG"}), encoding="utf-8")
    monkeypatch.setenv("BRAIN_DATA_DIR", str(data_dir))
    monkeypatch.delenv("BRAIN_LOG_LEVEL", raising=False)

    load_config()

    assert capsys.readouterr().err == ""
