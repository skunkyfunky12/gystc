import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from brain_mcp.daemon.server import build_guard_middleware, free_port

# NOTE: httpx 0.28.1's ASGITransport is async-only (subclass of AsyncBaseTransport,
# implements only handle_async_request), so it cannot be driven by the sync httpx.Client.
# We therefore exercise the ASGI app with httpx.AsyncClient + ASGITransport.

def _client(token, allowed_origins=("http://127.0.0.1",)):
    async def ok(request): return PlainTextResponse("ok")
    app = Starlette(routes=[Route("/mcp", ok, methods=["POST"])])
    app.add_middleware(build_guard_middleware(token=token, allowed_origins=list(allowed_origins)))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")

@pytest.mark.asyncio
async def test_rejects_without_token():
    async with _client("good") as c:
        assert (await c.post("/mcp")).status_code == 401

@pytest.mark.asyncio
async def test_accepts_with_token():
    async with _client("good") as c:
        assert (await c.post("/mcp", headers={"Authorization": "Bearer good"})).status_code == 200

@pytest.mark.asyncio
async def test_rejects_wrong_token():
    async with _client("good") as c:
        assert (await c.post("/mcp", headers={"Authorization": "Bearer bad"})).status_code == 401

@pytest.mark.asyncio
async def test_rejects_foreign_origin():
    async with _client("good", allowed_origins=["http://127.0.0.1"]) as c:
        r = await c.post("/mcp", headers={"Authorization": "Bearer good", "Origin": "http://evil.com"})
        assert r.status_code == 403

@pytest.mark.asyncio
async def test_allows_known_origin():
    async with _client("good", allowed_origins=["http://127.0.0.1"]) as c:
        r = await c.post("/mcp", headers={"Authorization": "Bearer good", "Origin": "http://127.0.0.1"})
        assert r.status_code == 200

@pytest.mark.asyncio
async def test_rejects_oversized_body():
    async def ok(request): return PlainTextResponse("ok")
    app = Starlette(routes=[Route("/mcp", ok, methods=["POST"])])
    app.add_middleware(build_guard_middleware(
        token="good", allowed_origins=["http://127.0.0.1"], max_body=50))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://testserver") as c:
        big = await c.post("/mcp", headers={"Authorization": "Bearer good"}, content=b"x" * 100)
        assert big.status_code == 413
        small = await c.post("/mcp", headers={"Authorization": "Bearer good"}, content=b"x" * 10)
        assert small.status_code == 200


def test_free_port_is_int_in_range():
    p = free_port()
    assert isinstance(p, int) and 1024 <= p <= 65535
