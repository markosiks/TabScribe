import asyncio

import httpx
from fastapi import FastAPI


def test_health_returns_backend_defaults(asgi_app: FastAPI) -> None:
    response = asyncio.run(_get_health(asgi_app))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["host"] == "127.0.0.1"
    assert body["port"] == 8765
    assert body["default_profile"] == "balanced"
    assert body["mode"] == "local-first"


async def _get_health(asgi_app: FastAPI) -> httpx.Response:
    transport = httpx.ASGITransport(app=asgi_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get("/health")
