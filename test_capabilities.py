"""Integration tester for the Calculator A2A agent capabilities.

Run the agent server first:

    uv run -m agents.CalculatorAgent

Then run:

    uv run python test_capabilities.py --url http://127.0.0.1:9999 --token local-dev-token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
}


@dataclass
class CheckResult:
    """Stores one capability check result for the final summary."""

    name: str
    ok: bool
    detail: str


@dataclass
class PushCapture:
    """In-memory webhook state used by the push-notification test."""

    notifications: list[dict[str, Any]] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)

    async def handler(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            payload = {"raw": (await request.body()).decode("utf-8", errors="replace")}
        self.notifications.append(
            {
                "token": request.headers.get("x-a2a-notification-token"),
                "payload": payload,
            }
        )
        self.event.set()
        return JSONResponse({"ok": True})


def _headers(token: str | None = None, stream: bool = False) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "A2A-Version": "1.0"}
    if stream:
        headers["Accept"] = "text/event-stream"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _rpc(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params or {},
    }


def _message_params(
    message: str,
    *,
    configuration: dict[str, Any] | None = None,
    parts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "message": {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_USER",
            "parts": parts or [{"text": message}],
        }
    }
    if configuration:
        params["configuration"] = configuration
    return params


async def _post_rpc(
    client: httpx.AsyncClient,
    url: str,
    method: str,
    params: dict[str, Any] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    response = await client.post(
        url,
        json=_rpc(method, params),
        headers=_headers(token),
    )
    response.raise_for_status()
    return response.json()


async def _wait_for_agent(client: httpx.AsyncClient, url: str, timeout: float) -> None:
    """Poll the public agent card until a launched local server is ready."""
    card_url = url.rstrip("/") + "/.well-known/agent-card.json"
    deadline = asyncio.get_running_loop().time() + timeout
    last_error: Exception | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            response = await client.get(card_url)
            if response.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        await asyncio.sleep(0.25)
    raise RuntimeError(f"agent did not become ready at {card_url}: {last_error}")


def _start_agent_process(url: str) -> subprocess.Popen:
    """Launch the calculator agent in a child process for end-to-end testing."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 9999)
    env = os.environ.copy()
    env.setdefault("CALCULATOR_AGENT_HOST", host)
    env.setdefault("CALCULATOR_AGENT_PORT", port)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agents.CalculatorAgent",
            "--host",
            host,
            "--port",
            port,
        ],
        env=env,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )


def _stop_agent_process(process: subprocess.Popen | None) -> None:
    """Terminate a child agent process started by this tester."""
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _result_task(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a Task from both v1.1 and v0.3-compatible response shapes."""
    result = data.get("result")
    if not isinstance(result, dict):
        return None
    if isinstance(result.get("task"), dict):
        return result["task"]
    if "id" in result and "status" in result:
        return result
    return None


def _artifact_text(task_or_event: dict[str, Any]) -> str:
    artifacts = task_or_event.get("artifacts") or []
    if not artifacts and "artifact" in task_or_event:
        artifacts = [task_or_event["artifact"]]
    texts: list[str] = []
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            if "text" in part:
                texts.append(part["text"])
    return "\n".join(texts)


async def check_agent_card(client: httpx.AsyncClient, url: str) -> tuple[CheckResult, dict]:
    """Fetch the public agent card and verify advertised capability fields."""
    card_url = url.rstrip("/") + "/.well-known/agent-card.json"
    response = await client.get(card_url)
    response.raise_for_status()
    card = response.json()
    caps = card.get("capabilities", {})
    ok = bool(caps.get("streaming")) and bool(caps.get("extendedAgentCard"))
    detail = (
        f"streaming={caps.get('streaming')}, "
        f"extendedAgentCard={caps.get('extendedAgentCard')}, "
        f"pushNotifications={caps.get('pushNotifications')}"
    )
    return CheckResult("public agent card", ok, detail), card


async def check_chat(
    client: httpx.AsyncClient,
    url: str,
    message: str,
    token: str | None,
) -> CheckResult:
    """Send a normal SendMessage request and require a terminal task response."""
    data = await _post_rpc(
        client,
        url,
        "SendMessage",
        _message_params(message),
        token,
    )
    task = _result_task(data)
    if not task:
        return CheckResult("chat", False, f"unexpected response: {data}")
    state = task.get("status", {}).get("state")
    text = _artifact_text(task)
    return CheckResult("chat", state == "TASK_STATE_COMPLETED" and bool(text), text or state)


async def check_stream(
    client: httpx.AsyncClient,
    url: str,
    message: str,
    token: str | None,
    expect_opaque: bool,
    timeout: float,
) -> CheckResult:
    """Open SendStreamingMessage and inspect intermediate/final event shape."""
    events: list[dict[str, Any]] = []
    payload = _rpc("SendStreamingMessage", _message_params(message))
    async with client.stream(
        "POST",
        url,
        json=payload,
        headers=_headers(token, stream=True),
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            event = json.loads(raw)
            events.append(event)
            result = event.get("result", {})
            status_update = result.get("statusUpdate") or result.get("status_update")
            if status_update:
                state = status_update.get("status", {}).get("state")
                if state in TERMINAL_STATES:
                    break

    working_messages = [
        event
        for event in events
        if (
            (event.get("result", {}).get("statusUpdate") or {})
            .get("status", {})
            .get("state")
            == "TASK_STATE_WORKING"
        )
    ]
    artifact_texts = [
        _artifact_text(event.get("result", {}).get("artifactUpdate", {}))
        for event in events
    ]
    final_text = "\n".join(text for text in artifact_texts if text)
    opaque_ok = not working_messages if expect_opaque else bool(working_messages)
    ok = bool(events) and bool(final_text) and opaque_ok
    detail = (
        f"events={len(events)}, working_updates={len(working_messages)}, "
        f"artifact={final_text!r}"
    )
    return CheckResult("streaming" + (" opaque" if expect_opaque else ""), ok, detail)


async def check_extended_card(
    client: httpx.AsyncClient,
    url: str,
    token: str | None,
) -> CheckResult:
    """Fetch the authenticated extended card when a bearer token is supplied."""
    if not token:
        return CheckResult("extended card", False, "skipped; provide --token")
    data = await _post_rpc(client, url, "GetExtendedAgentCard", {}, token)
    card = data.get("result", {})
    skill_names = [skill.get("name") for skill in card.get("skills", [])]
    ok = any("Advanced" in (name or "") for name in skill_names)
    return CheckResult("extended card", ok, f"skills={skill_names}")


async def check_task_management(
    client: httpx.AsyncClient,
    url: str,
    message: str,
    token: str | None,
) -> CheckResult:
    """Exercise SendMessage(returnImmediately), GetTask, ListTasks, and CancelTask."""
    data = await _post_rpc(
        client,
        url,
        "SendMessage",
        _message_params(
            message,
            configuration={"returnImmediately": True, "historyLength": 10},
        ),
        token,
    )
    task = _result_task(data)
    if not task:
        return CheckResult("task management", False, f"no task returned: {data}")

    task_id = task["id"]
    context_id = task.get("contextId") or task.get("context_id")

    get_task: dict[str, Any] = {}
    listed: dict[str, Any] = {}
    got: dict[str, Any] | None = None
    listed_count = 0
    deadline = asyncio.get_running_loop().time() + 10.0
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.5)
        get_task = await _post_rpc(
            client,
            url,
            "GetTask",
            {"id": task_id, "history_length": 10},
            token,
        )
        listed = await _post_rpc(
            client,
            url,
            "ListTasks",
            {"context_id": context_id, "page_size": 10}
            if context_id
            else {"page_size": 10},
            token,
        )
        got = _result_task(get_task)
        list_result = listed.get("result", {})
        listed_count = len(list_result.get("tasks", []))
        if got and listed_count >= 1:
            break

    cancel = await _post_rpc(client, url, "CancelTask", {"id": task_id}, token)

    cancel_code = cancel.get("error", {}).get("code")
    ok = bool(got) and listed_count >= 1 and cancel_code in {-32001, -32002}
    detail = (
        f"task_id={task_id}, listed={listed_count}, cancel_code={cancel_code}"
    )
    if not ok:
        detail += (
            f", get_error={get_task.get('error')}, "
            f"list_error={listed.get('error')}, cancel_error={cancel.get('error')}"
        )
    return CheckResult(
        "task management",
        ok,
        detail,
    )


async def check_push_notifications(
    url: str,
    message: str,
    token: str | None,
    webhook_host: str,
    webhook_port: int,
    timeout: float,
) -> CheckResult:
    """Start a local webhook and request inline push notifications from the agent."""
    capture = PushCapture()
    webhook_url = f"http://{webhook_host}:{webhook_port}/notify"
    webhook_app = Starlette(routes=[Route("/notify", capture.handler, methods=["POST"])])
    server = uvicorn.Server(
        uvicorn.Config(webhook_app, host=webhook_host, port=webhook_port, log_level="warning")
    )
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            data = await _post_rpc(
                client,
                url,
                "SendMessage",
                _message_params(
                    message,
                    configuration={
                        "returnImmediately": True,
                        "taskPushNotificationConfig": {
                            "id": "capability-test",
                            "url": webhook_url,
                            "token": "capability-test-token",
                        },
                    },
                ),
                token,
            )
            if "error" in data:
                return CheckResult("push notifications", False, json.dumps(data["error"]))

            try:
                await asyncio.wait_for(capture.event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return CheckResult(
                    "push notifications",
                    False,
                    "no webhook received; task may have completed before registration",
                )

            token_ok = any(
                item.get("token") == "capability-test-token"
                for item in capture.notifications
            )
            return CheckResult(
                "push notifications",
                token_ok,
                f"received={len(capture.notifications)}, token_ok={token_ok}",
            )
    finally:
        server.should_exit = True
        await server_task


async def check_multimodal(
    client: httpx.AsyncClient,
    url: str,
    card: dict[str, Any],
    token: str | None,
) -> CheckResult:
    """Send a structured data part only when the card advertises JSON input."""
    input_modes = set(card.get("defaultInputModes") or card.get("default_input_modes") or [])
    if "application/json" not in input_modes:
        return CheckResult("multi-modal", True, "skipped; card does not advertise application/json")

    data = await _post_rpc(
        client,
        url,
        "SendMessage",
        _message_params(
            "",
            parts=[
                {"text": "Use this JSON object to answer: "},
                {"data": {"expression": "2+3*5", "expected": 17}, "mediaType": "application/json"},
            ],
        ),
        token,
    )
    task = _result_task(data)
    ok = bool(task) and task.get("status", {}).get("state") == "TASK_STATE_COMPLETED"
    return CheckResult("multi-modal", ok, _artifact_text(task or {}) or json.dumps(data))


def _print_result(result: CheckResult) -> None:
    marker = "PASS" if result.ok else "FAIL"
    print(f"[{marker}] {result.name}: {result.detail}")


async def run(args: argparse.Namespace) -> int:
    """Run all requested checks and return a process exit code."""
    process: subprocess.Popen | None = None
    results: list[CheckResult] = []
    try:
        if args.start_agent:
            process = _start_agent_process(args.url)

        async with httpx.AsyncClient(timeout=args.timeout) as client:
            if args.start_agent:
                await _wait_for_agent(client, args.url, args.timeout)

            card_result, card = await check_agent_card(client, args.url)
            results.append(card_result)
            results.append(await check_chat(client, args.url, args.message, args.token))
            results.append(
                await check_stream(
                    client,
                    args.url,
                    args.message,
                    args.token,
                    args.expect_opaque,
                    args.timeout,
                )
            )
            results.append(await check_extended_card(client, args.url, args.token))
            results.append(await check_task_management(client, args.url, args.message, args.token))
            results.append(await check_multimodal(client, args.url, card, args.token))

        if not args.skip_push:
            results.append(
                await check_push_notifications(
                    args.url,
                    args.message,
                    args.token,
                    args.webhook_host,
                    args.webhook_port,
                    args.timeout,
                )
            )

        print("\nCapability test summary")
        print("-" * 80)
        for result in results:
            _print_result(result)

        return 0 if all(result.ok for result in results) else 1
    finally:
        _stop_agent_process(process)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Calculator A2A capabilities.")
    parser.add_argument("--url", default="http://127.0.0.1:9999")
    parser.add_argument("--message", default="What is 2+3*5?")
    parser.add_argument("--token", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--start-agent",
        action="store_true",
        help="Launch agents.CalculatorAgent before running checks.",
    )
    parser.add_argument(
        "--expect-opaque",
        dest="expect_opaque",
        action="store_true",
        default=True,
        help="Require streaming output to hide WORKING events.",
    )
    parser.add_argument(
        "--expect-working",
        dest="expect_opaque",
        action="store_false",
        help="Require streaming output to include WORKING events.",
    )
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--webhook-host", default="127.0.0.1")
    parser.add_argument("--webhook-port", type=int, default=8765)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
