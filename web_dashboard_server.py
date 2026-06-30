"""
A2A Web Dashboard relay server.

The browser talks to this local FastAPI-compatible Starlette relay instead of
calling the agent directly. The relay avoids CORS issues, forwards bearer
tokens only server-side, and hosts the push-notification webhook used by the UI.

Run:
    python web_dashboard_server.py
    # then open http://127.0.0.1:8766 in a browser
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import click
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard.html"
NOTIFY_PATH = "/notify"

# Local dev storage for push notifications and SSE browser subscribers.
_subscribers: list[asyncio.Queue] = []
_notification_log: list[dict] = []
_log_lock = asyncio.Lock()


def _now_iso() -> str:
    """Return a compact UTC timestamp for notification records."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


async def _broadcast(event: dict) -> None:
    """Persist a notification and fan it out to connected dashboard tabs."""
    async with _log_lock:
        _notification_log.append(event)
        if len(_notification_log) > 200:
            _notification_log.pop(0)
    for queue in list(_subscribers):
        await queue.put(event)


async def index(request: Request) -> HTMLResponse:
    """Serve the single-page dashboard UI."""
    if not DASHBOARD_HTML_PATH.exists():
        return HTMLResponse(
            "<h1>dashboard.html not found</h1>"
            f"<p>Expected at: {DASHBOARD_HTML_PATH}</p>",
            status_code=500,
        )
    return HTMLResponse(DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))


async def webhook_url(request: Request) -> JSONResponse:
    """Return the relay webhook URL that agents should call for push updates."""
    host = request.url.hostname or "127.0.0.1"
    port = request.url.port or 8766
    return JSONResponse({"url": f"http://{host}:{port}{NOTIFY_PATH}"})


async def receive_notification(request: Request) -> JSONResponse:
    """Receive agent push notifications and publish them to the browser UI."""
    try:
        payload = await request.json()
    except Exception:
        raw = await request.body()
        payload = {"_raw": raw.decode("utf-8", errors="replace")}

    event = {
        "id": str(uuid.uuid4()),
        "receivedAt": _now_iso(),
        "headers": {k: v for k, v in request.headers.items() if k.lower() != "authorization"},
        "payload": payload,
    }
    await _broadcast(event)
    return JSONResponse({"status": "received"})


async def events_stream(request: Request) -> StreamingResponse:
    """Stream push notifications to browser tabs using Server-Sent Events."""
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.append(queue)

    async def gen() -> AsyncIterator[bytes]:
        try:
            async with _log_lock:
                backlog = list(_notification_log)
            for event in backlog:
                yield f"data: {json.dumps(event)}\n\n".encode()

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(event)}\n\n".encode()
                except asyncio.CancelledError:
                    break
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
        finally:
            if queue in _subscribers:
                _subscribers.remove(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def clear_notifications(request: Request) -> JSONResponse:
    """Clear the relay-side push notification history."""
    async with _log_lock:
        _notification_log.clear()
    return JSONResponse({"status": "cleared"})


async def proxy_agent_card(request: Request) -> JSONResponse:
    """Fetch the public agent card for the user-provided A2A URL."""
    agent_url = request.query_params.get("agentUrl", "").rstrip("/")
    if not agent_url:
        return JSONResponse({"error": "agentUrl query param required"}, status_code=400)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{agent_url}/.well-known/agent-card.json")
            response.raise_for_status()
            return JSONResponse(response.json())
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"error": f"Could not reach agent at {agent_url}: {exc}"},
                status_code=502,
            )


async def proxy_rpc(request: Request) -> Response:
    """Forward dashboard JSON-RPC calls to the selected A2A agent."""
    body = await request.json()
    agent_url = body.get("agentUrl", "").rstrip("/")
    token = body.get("token") or None
    method = body.get("method")
    params = body.get("params", {})
    streaming = bool(body.get("streaming", False))

    if not agent_url or not method:
        return JSONResponse({"error": "agentUrl and method are required"}, status_code=400)

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }
    headers = {"Content-Type": "application/json", "A2A-Version": "1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if streaming:
        headers["Accept"] = "text/event-stream"

        async def gen() -> AsyncIterator[bytes]:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", agent_url, json=payload, headers=headers) as response:
                        if response.status_code >= 400:
                            err_body = await response.aread()
                            err = json.dumps(
                                {
                                    "error": {
                                        "code": response.status_code,
                                        "message": err_body.decode("utf-8", "replace"),
                                    }
                                }
                            )
                            yield f"data: {err}\n\n".encode()
                            return
                        async for chunk in response.aiter_raw():
                            yield chunk
            except asyncio.CancelledError:
                return
            except Exception as exc:
                err = json.dumps({"error": {"message": f"Relay error: {exc}"}})
                yield f"data: {err}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(agent_url, json=payload, headers=headers)
            try:
                return JSONResponse(response.json(), status_code=response.status_code)
            except json.JSONDecodeError:
                return JSONResponse(
                    {"error": f"Non-JSON response (status {response.status_code})"},
                    status_code=502,
                )
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"error": {"message": f"Could not reach agent: {exc}"}},
                status_code=502,
            )


routes = [
    Route("/", index, methods=["GET"]),
    Route("/api/webhook-url", webhook_url, methods=["GET"]),
    Route("/api/agent-card", proxy_agent_card, methods=["GET"]),
    Route("/api/rpc", proxy_rpc, methods=["POST"]),
    Route("/api/notifications", clear_notifications, methods=["DELETE"]),
    Route("/events", events_stream, methods=["GET"]),
    Route(NOTIFY_PATH, receive_notification, methods=["POST"]),
]

app = Starlette(routes=routes)


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8766, show_default=True, type=int)
def main(host: str, port: int) -> None:
    """Start the dashboard relay server."""
    print("\n  A2A Web Dashboard")
    print(f"  Open:    http://{host}:{port}")
    print(f"  Webhook: http://{host}:{port}{NOTIFY_PATH}  (auto-filled in the Push tab)\n")
    uvicorn.run(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
