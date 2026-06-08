import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from brain_mcp.daemon.proxy import forward_one

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
            self.send_response(202); self.end_headers()
        def log_message(self,*a): pass
    srv = _serve(H); port = srv.server_address[1]
    try:
        assert forward_one(f"http://127.0.0.1:{port}/mcp", "tok",
                           {"jsonrpc":"2.0","method":"notifications/initialized"}) is None
    finally:
        srv.shutdown()
