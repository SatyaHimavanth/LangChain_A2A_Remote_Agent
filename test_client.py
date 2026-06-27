from __future__ import annotations

import argparse
import asyncio
import uuid

import httpx


def _build_request(message: str, response_type: str) -> dict:
    method = "SendStreamingMessage" if response_type == "stream" else "SendMessage"
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "ROLE_USER",
                "parts": [{"text": message}],
            }
        },
    }


def _build_headers(token: str | None, response_type: str) -> dict[str, str]:
    headers = {"A2A-Version": "1.0"}
    if response_type == "stream":
        headers["Accept"] = "text/event-stream"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def send_chat_message(url: str, message: str, token: str | None, timeout: float) -> None:
    headers = _build_headers(token, "chat")
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            json=_build_request(message, "chat"),
            headers=headers,
        )
        response.raise_for_status()
        print(response.text)


async def send_streaming_message(url: str, message: str, token: str | None, timeout: float) -> None:
    headers = _build_headers(token, "stream")
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            url,
            json=_build_request(message, "stream"),
            headers=headers,
        ) as response:
            response.raise_for_status()
            event_lines: list[str] = []

            async for line in response.aiter_lines():
                if not line:
                    if event_lines:
                        print("\n".join(event_lines))
                        event_lines.clear()
                    continue

                if line.startswith("data:"):
                    event_lines.append(line.removeprefix("data:").strip())
                elif line.startswith("event:"):
                    event_lines.append(line.strip())

            if event_lines:
                print("\n".join(event_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a JSON-RPC message to a running A2A agent.")
    parser.add_argument("--url", default="http://127.0.0.1:9999", help="Agent JSON-RPC endpoint.")
    parser.add_argument("--message", required=True, help="Message to send to the agent.")
    parser.add_argument("--token", default=None, help="Optional bearer token for authenticated skills.")
    parser.add_argument("--timeout", default=180.0, type=float, help="HTTP read timeout in seconds.")
    parser.add_argument(
        "--response-type",
        choices=["chat", "stream"],
        default="chat",
        help="Use chat for SendMessage or stream for SendStreamingMessage.",
    )
    args = parser.parse_args()

    if args.response_type == "stream":
        asyncio.run(send_streaming_message(args.url, args.message, args.token, args.timeout))
    else:
        asyncio.run(send_chat_message(args.url, args.message, args.token, args.timeout))


if __name__ == "__main__":
    main()
