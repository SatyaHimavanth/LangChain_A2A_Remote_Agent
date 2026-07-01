"""Unified CLI test client for A2A calculator-agent capabilities."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

console = Console()

ALL_TESTS = [
    "card",
    "chat",
    "stream",
    "extended",
    "auth",
    "tasks",
    "multimodal",
    "content-type",
    "push",
]
TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_AUTH_REQUIRED",
}


@dataclass(slots=True)
class TestResult:
    """One CLI test outcome."""

    name: str
    ok: bool
    detail: str
    duration_ms: float
    raw: dict[str, Any] | list[dict[str, Any]] | None = None


@dataclass
class PushCapture:
    """Local webhook state for push notification tests."""

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
    """Poll the public agent card until a local child server is ready."""
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
    """Launch the local CalculatorAgent for self-contained test runs."""
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
    """Stop a child server started by this client."""
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


def _task_from_result(data: dict[str, Any]) -> dict[str, Any] | None:
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
    if not artifacts and "artifactUpdate" in task_or_event:
        artifacts = [task_or_event["artifactUpdate"].get("artifact", {})]
    texts: list[str] = []
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            if "text" in part:
                texts.append(part["text"])
    return "\n".join(texts)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class A2ACapabilityClient:
    """Runs selected capability checks against one A2A agent URL."""

    def __init__(self, url: str, token: str | None, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.card: dict[str, Any] | None = None

    async def check_card(self, client: httpx.AsyncClient) -> TestResult:
        started = time.perf_counter()
        response = await client.get(f"{self.url}/.well-known/agent-card.json")
        response.raise_for_status()
        self.card = response.json()
        caps = self.card.get("capabilities", {})
        input_modes = self.card.get("defaultInputModes") or []
        ok = bool(caps.get("streaming")) and bool(caps.get("extendedAgentCard"))
        detail = (
            f"{self.card.get('name', 'agent')} v{self.card.get('version', '?')}; "
            f"streaming={caps.get('streaming')}; "
            f"push={caps.get('pushNotifications')}; "
            f"inputs={', '.join(input_modes) or 'unknown'}"
        )
        return TestResult("Agent card", ok, detail, _elapsed_ms(started), self.card)

    async def check_chat(self, client: httpx.AsyncClient, message: str) -> TestResult:
        started = time.perf_counter()
        data = await _post_rpc(client, self.url, "SendMessage", _message_params(message), self.token)
        task = _task_from_result(data)
        if not task:
            return TestResult("Chat invoke", False, f"unexpected response: {data}", _elapsed_ms(started), data)
        state = task.get("status", {}).get("state")
        text = _artifact_text(task)
        ok = state == "TASK_STATE_COMPLETED" and bool(text)
        return TestResult("Chat invoke", ok, f"state={state}; artifact={text!r}", _elapsed_ms(started), data)

    async def check_stream(
        self,
        client: httpx.AsyncClient,
        message: str,
        expect_opaque: bool | None,
    ) -> TestResult:
        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        payload = _rpc("SendStreamingMessage", _message_params(message))
        async with client.stream(
            "POST",
            self.url,
            json=payload,
            headers=_headers(self.token, stream=True),
            timeout=self.timeout,
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
                status_update = event.get("result", {}).get("statusUpdate") or {}
                state = status_update.get("status", {}).get("state")
                if state in TERMINAL_STATES:
                    break

        working = [
            event
            for event in events
            if (
                (event.get("result", {}).get("statusUpdate") or {})
                .get("status", {})
                .get("state")
                == "TASK_STATE_WORKING"
            )
        ]
        artifacts = [
            _artifact_text(event.get("result", {}).get("artifactUpdate", {}))
            for event in events
        ]
        artifact_text = "\n".join(text for text in artifacts if text)
        if expect_opaque is True:
            mode_ok = len(working) == 0
        elif expect_opaque is False:
            mode_ok = len(working) > 0
        else:
            mode_ok = True
        ok = bool(events) and bool(artifact_text) and mode_ok
        detail = f"events={len(events)}; working_updates={len(working)}; artifact={artifact_text!r}"
        return TestResult("Streaming", ok, detail, _elapsed_ms(started), events)

    async def check_extended_card(self, client: httpx.AsyncClient) -> TestResult:
        started = time.perf_counter()
        if not self.token:
            return TestResult("Extended card", False, "token required", _elapsed_ms(started))
        data = await _post_rpc(client, self.url, "GetExtendedAgentCard", {}, self.token)
        card = data.get("result", {})
        skills = [skill.get("name") for skill in card.get("skills", [])]
        ok = any("Advanced" in (name or "") for name in skills)
        return TestResult("Extended card", ok, f"skills={skills}", _elapsed_ms(started), data)

    async def check_auth_required(self, client: httpx.AsyncClient) -> TestResult:
        started = time.perf_counter()
        data = await _post_rpc(client, self.url, "SendMessage", _message_params("What is sqrt(16)?"), None)
        task = _task_from_result(data)
        state = (task or {}).get("status", {}).get("state")
        ok = state == "TASK_STATE_AUTH_REQUIRED"
        return TestResult("Auth required state", ok, f"state={state}", _elapsed_ms(started), data)

    async def check_tasks(self, client: httpx.AsyncClient, message: str) -> TestResult:
        started = time.perf_counter()
        data = await _post_rpc(
            client,
            self.url,
            "SendMessage",
            _message_params(
                message,
                configuration={"returnImmediately": True, "historyLength": 10},
            ),
            self.token,
        )
        task = _task_from_result(data)
        if not task:
            return TestResult("Task management", False, f"no task returned: {data}", _elapsed_ms(started), data)

        task_id = task["id"]
        context_id = task.get("contextId") or task.get("context_id")
        got: dict[str, Any] | None = None
        listed_count = 0
        get_task: dict[str, Any] = {}
        listed: dict[str, Any] = {}
        deadline = asyncio.get_running_loop().time() + min(self.timeout, 12.0)
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.5)
            get_task = await _post_rpc(
                client,
                self.url,
                "GetTask",
                {"id": task_id, "history_length": 10},
                self.token,
            )
            listed = await _post_rpc(
                client,
                self.url,
                "ListTasks",
                {"context_id": context_id, "page_size": 10} if context_id else {"page_size": 10},
                self.token,
            )
            got = _task_from_result(get_task)
            listed_count = len((listed.get("result") or {}).get("tasks", []))
            if got and listed_count >= 1:
                break

        cancel = await _post_rpc(client, self.url, "CancelTask", {"id": task_id}, self.token)
        cancel_code = (cancel.get("error") or {}).get("code")
        ok = bool(got) and listed_count >= 1 and cancel_code in {-32001, -32002}
        detail = f"task_id={task_id}; listed={listed_count}; cancel_code={cancel_code}"
        if not ok:
            detail += f"; get_error={get_task.get('error')}; list_error={listed.get('error')}"
        return TestResult("Task management", ok, detail, _elapsed_ms(started), {"get": get_task, "list": listed, "cancel": cancel})

    async def check_multimodal(self, client: httpx.AsyncClient) -> TestResult:
        started = time.perf_counter()
        if self.card is None:
            await self.check_card(client)
        input_modes = set((self.card or {}).get("defaultInputModes") or [])
        if "application/json" not in input_modes:
            return TestResult("Multimodal", True, "skipped, application/json not advertised", _elapsed_ms(started))

        data = await _post_rpc(
            client,
            self.url,
            "SendMessage",
            _message_params(
                "",
                parts=[
                    {"text": "Use this JSON object to answer: "},
                    {"data": {"expression": "2+3*5", "expected": 17}, "mediaType": "application/json"},
                ],
            ),
            self.token,
        )
        task = _task_from_result(data)
        state = (task or {}).get("status", {}).get("state")
        text = _artifact_text(task or {})
        ok = bool(task) and state == "TASK_STATE_COMPLETED" and bool(text)
        return TestResult("Multimodal", ok, f"state={state}; artifact={text!r}", _elapsed_ms(started), data)

    async def check_content_type_validation(self, client: httpx.AsyncClient) -> TestResult:
        started = time.perf_counter()
        data = await _post_rpc(
            client,
            self.url,
            "SendMessage",
            _message_params(
                "",
                parts=[
                    {"text": "Please process this unsupported file."},
                    {
                        "raw": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA",
                        "mediaType": "audio/wav",
                        "filename": "tone.wav",
                    },
                ],
            ),
            self.token,
        )
        error = data.get("error") or {}
        code = error.get("code")
        ok = code == -32005
        return TestResult("Content-type validation", ok, f"error_code={code}", _elapsed_ms(started), data)

    async def check_push(
        self,
        message: str,
        host: str,
        port: int,
    ) -> TestResult:
        started = time.perf_counter()
        capture = PushCapture()
        webhook_url = f"http://{host}:{port}/notify"
        webhook_app = Starlette(routes=[Route("/notify", capture.handler, methods=["POST"])])
        server = uvicorn.Server(
            uvicorn.Config(webhook_app, host=host, port=port, log_level="warning")
        )
        server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.4)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = await _post_rpc(
                    client,
                    self.url,
                    "SendMessage",
                    _message_params(
                        message,
                        configuration={
                            "returnImmediately": True,
                            "taskPushNotificationConfig": {
                                "id": "cli-capability-test",
                                "url": webhook_url,
                                "token": "cli-capability-token",
                            },
                        },
                    ),
                    self.token,
                )
                if "error" in data:
                    return TestResult("Push notifications", False, json.dumps(data["error"]), _elapsed_ms(started), data)
                task = _task_from_result(data)
                task_id = (task or {}).get("id")
                if not task_id:
                    return TestResult("Push notifications", False, "no task id returned", _elapsed_ms(started), data)

                try:
                    await asyncio.wait_for(capture.event.wait(), timeout=self.timeout)
                except asyncio.TimeoutError:
                    return TestResult("Push notifications", False, "no webhook delivery received", _elapsed_ms(started), data)

                token_ok = any(item.get("token") == "cli-capability-token" for item in capture.notifications)
                get_config = await _post_rpc(
                    client,
                    self.url,
                    "GetTaskPushNotificationConfig",
                    {"task_id": task_id, "id": "cli-capability-test"},
                    self.token,
                )
                list_configs = await _post_rpc(
                    client,
                    self.url,
                    "ListTaskPushNotificationConfigs",
                    {"task_id": task_id, "page_size": 10},
                    self.token,
                )
                delete_config = await _post_rpc(
                    client,
                    self.url,
                    "DeleteTaskPushNotificationConfig",
                    {"task_id": task_id, "id": "cli-capability-test"},
                    self.token,
                )
                config_ok = (
                    not get_config.get("error")
                    and not list_configs.get("error")
                    and not delete_config.get("error")
                    and bool((list_configs.get("result") or {}).get("configs"))
                )
                detail = (
                    f"received={len(capture.notifications)}; token_ok={token_ok}; "
                    f"config_management={config_ok}"
                )
                raw = {
                    "notifications": capture.notifications,
                    "get": get_config,
                    "list": list_configs,
                    "delete": delete_config,
                }
                return TestResult("Push notifications", token_ok and config_ok, detail, _elapsed_ms(started), raw)
        finally:
            server.should_exit = True
            await server_task


async def _run_selected(args: argparse.Namespace, tests: list[str]) -> int:
    process: subprocess.Popen | None = None
    results: list[TestResult] = []
    runner = A2ACapabilityClient(args.agent, args.token, args.timeout)
    try:
        if args.start_agent:
            process = _start_agent_process(args.agent)

        async with httpx.AsyncClient(timeout=args.timeout) as client:
            if args.start_agent:
                await _wait_for_agent(client, args.agent, args.timeout)

            for name in tests:
                if name == "card":
                    results.append(await runner.check_card(client))
                elif name == "chat":
                    results.append(await runner.check_chat(client, args.message))
                elif name == "stream":
                    results.append(await runner.check_stream(client, args.message, args.expect_opaque))
                elif name == "extended":
                    results.append(await runner.check_extended_card(client))
                elif name == "auth":
                    results.append(await runner.check_auth_required(client))
                elif name == "tasks":
                    results.append(await runner.check_tasks(client, args.message))
                elif name == "multimodal":
                    results.append(await runner.check_multimodal(client))
                elif name == "content-type":
                    results.append(await runner.check_content_type_validation(client))
                elif name == "push":
                    results.append(await runner.check_push(args.message, args.webhook_host, args.webhook_port))

                _print_result(results[-1], args.verbose)

        _print_summary(results)
        return 0 if all(result.ok for result in results) else 1
    finally:
        _stop_agent_process(process)


def _print_result(result: TestResult, verbose: bool) -> None:
    status = "[green]PASS[/]" if result.ok else "[red]FAIL[/]"
    console.print(
        Panel(
            result.detail,
            title=f"{status} {result.name} [{result.duration_ms:.0f} ms]",
            border_style="green" if result.ok else "red",
        )
    )
    if verbose and result.raw is not None:
        console.print_json(json.dumps(result.raw))


def _print_summary(results: list[TestResult]) -> None:
    table = Table(title="Capability Test Summary")
    table.add_column("Capability")
    table.add_column("Result", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Detail")
    for result in results:
        table.add_row(
            result.name,
            "PASS" if result.ok else "FAIL",
            f"{result.duration_ms:.0f} ms",
            result.detail,
        )
    console.print(table)


def _parse_tests(raw: str) -> list[str]:
    if raw == "all":
        return list(ALL_TESTS)
    selected = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in ALL_TESTS]
    if unknown:
        raise SystemExit(f"Unknown test(s): {', '.join(unknown)}. Choices: {', '.join(ALL_TESTS)}")
    return selected


def _select_tests_interactively() -> list[str]:
    console.print("[bold]Select capabilities to test[/]")
    selected: list[str] = []
    for name in ALL_TESTS:
        if Confirm.ask(f"Run {name}?", default=True):
            selected.append(name)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified A2A CLI capability test client.")
    parser.add_argument("--agent", default="http://127.0.0.1:9999", help="A2A JSON-RPC endpoint.")
    parser.add_argument("--token", default=None, help="Bearer token for extended/authenticated tests.")
    parser.add_argument("--tests", default="all", help="all or comma-separated names: " + ", ".join(ALL_TESTS))
    parser.add_argument("--select", action="store_true", help="Interactively choose tests with Rich prompts.")
    parser.add_argument("--message", default="What is 2+3*5?", help="Message used by chat, stream, task, and push tests.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--start-agent", action="store_true", help="Launch agents.CalculatorAgent before testing.")
    parser.add_argument("--expect-opaque", dest="expect_opaque", action="store_true", default=True)
    parser.add_argument("--expect-working", dest="expect_opaque", action="store_false")
    parser.add_argument("--webhook-host", default="127.0.0.1")
    parser.add_argument("--webhook-port", type=int, default=8765)
    parser.add_argument("--verbose", action="store_true", help="Print raw JSON for each check.")
    args = parser.parse_args()

    tests = _select_tests_interactively() if args.select else _parse_tests(args.tests)
    if not tests:
        raise SystemExit("No tests selected.")
    raise SystemExit(asyncio.run(_run_selected(args, tests)))


if __name__ == "__main__":
    main()
