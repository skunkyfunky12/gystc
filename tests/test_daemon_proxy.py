import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from brain_mcp.daemon.proxy import _force_utf8_stdio, forward_one

def _serve(handler):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

def test_forward_returns_json_response():
    captured = {}
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["Content-Length"]); body = self.rfile.read(n)
            captured["auth"] = self.headers.get("Authorization")
            captured["body"] = json.loads(body)
            resp = json.dumps({"jsonrpc":"2.0","id":1,"result":{"ok":True}}).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length", str(len(resp))); self.end_headers()
            self.wfile.write(resp)
        def log_message(self,*a): pass
    srv = _serve(H); port = srv.server_address[1]
    try:
        out = forward_one(f"http://127.0.0.1:{port}/mcp", "tok",
                          {"jsonrpc":"2.0","id":1,"method":"tools/list"})
        assert captured["auth"] == "Bearer tok"
        assert out == {"jsonrpc":"2.0","id":1,"result":{"ok":True}}
    finally:
        srv.shutdown()

def test_forward_returns_none_on_202():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            # Read the body before answering: BaseHTTPRequestHandler closes the
            # connection at the end of the request, and on BSD/macOS a socket
            # still holding unread received data sends an RST instead of a FIN,
            # so the client raises ECONNRESET instead of seeing the 202. The
            # same shape was a real bug in brain/web_widget.py's reject paths.
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            self.send_response(202); self.end_headers()
        def log_message(self,*a): pass
    srv = _serve(H); port = srv.server_address[1]
    try:
        assert forward_one(f"http://127.0.0.1:{port}/mcp", "tok",
                           {"jsonrpc":"2.0","method":"notifications/initialized"}) is None
    finally:
        srv.shutdown()

def test_force_utf8_stdio_fixes_cp1252_stdin(monkeypatch):
    # Windows without PYTHONUTF8=1: piped stdio defaults to cp1252, but MCP
    # traffic is UTF-8 -- without the reconfigure, umlauts/em-dash mojibake.
    raw = '{"msg":"Grüße äöüß"}\n'.encode()
    fake_stdin = io.TextIOWrapper(io.BytesIO(raw), encoding="cp1252")
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    _force_utf8_stdio()
    assert json.loads(sys.stdin.readline())["msg"] == "Grüße äöüß"


def test_force_utf8_stdio_skips_objects_without_reconfigure(monkeypatch):
    # pytest replaces stdin/stdout with capture objects lacking reconfigure() --
    # the hasattr guard must no-op instead of raising AttributeError.
    class NoReconfigure:
        pass

    monkeypatch.setattr(sys, "stdin", NoReconfigure())
    monkeypatch.setattr(sys, "stdout", NoReconfigure())
    _force_utf8_stdio()  # must not raise


def test_recovery_emits_jsonrpc_error_for_request_when_unreachable(monkeypatch):
    import brain_mcp.daemon.proxy as proxy
    import httpx
    monkeypatch.setattr(proxy, "ensure_daemon", lambda data_dir: None)  # cannot respawn
    state = {"info": type("I", (), {"token": "t"})(), "url": "http://127.0.0.1:9/mcp"}  # port 9 = dead
    with httpx.Client(timeout=2.0) as c:
        req = proxy._forward_with_recovery(c, state, None, {"jsonrpc":"2.0","id":7,"method":"tools/list"})
        assert req["id"] == 7 and req["error"]["code"] == -32000
        note = proxy._forward_with_recovery(c, state, None, {"jsonrpc":"2.0","method":"notifications/x"})
        assert note is None
