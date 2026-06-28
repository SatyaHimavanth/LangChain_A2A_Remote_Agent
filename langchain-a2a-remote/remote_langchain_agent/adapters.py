"""
adapters.py
~~~~~~~~~~~
Bidirectional conversion between LangChain message types and the A2A
protocol types from ``a2a-sdk >= 1.1.0`` (protobuf-based).

All A2A types are native protobuf messages from ``a2a.types``.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from .exceptions import A2AProtocolError, InputNormalisationError

try:
    from a2a.types import (
        Artifact,
        Message as A2AMessage,
        Part as A2APart,
        Role as A2ARole,
        SendMessageRequest,
        Task as A2ATask,
        TaskArtifactUpdateEvent,
        TaskState,
        TaskStatus,
        TaskStatusUpdateEvent,
    )
    _A2A_AVAILABLE = True
except ImportError:
    _A2A_AVAILABLE = False


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

def normalise_input(raw: Any) -> tuple[list[BaseMessage], dict[str, Any]]:
    """Coerce the many accepted input shapes into a canonical message list."""
    extra: dict[str, Any] = {}

    if isinstance(raw, str):
        return [HumanMessage(content=raw)], extra
    if isinstance(raw, HumanMessage):
        return [raw], extra
    if isinstance(raw, list):
        if not raw:
            raise InputNormalisationError("Input list is empty.")
        if not all(isinstance(m, BaseMessage) for m in raw):
            raise InputNormalisationError("List must contain only BaseMessage instances.")
        return raw, extra
    if isinstance(raw, dict):
        messages_raw = raw.get("messages")
        if messages_raw is None:
            raise InputNormalisationError('Input dict must have a "messages" key.')
        messages: list[BaseMessage] = []
        for item in messages_raw:
            if isinstance(item, BaseMessage):
                messages.append(item)
            elif isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
            else:
                raise InputNormalisationError(f"Unexpected item type: {type(item)}")
        extra = {k: v for k, v in raw.items() if k != "messages"}
        return messages, extra

    raise InputNormalisationError(
        f"Cannot normalise input of type {type(raw).__name__}."
    )


# ---------------------------------------------------------------------------
# LangChain → A2A (protobuf)
# ---------------------------------------------------------------------------

def _content_to_text(content: Any) -> str:
    """Extract plain text from a LangChain message content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def lc_messages_to_a2a_message(
    messages: list[BaseMessage],
    *,
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> "SendMessageRequest":
    """Convert a LangChain message list into an A2A SendMessageRequest (protobuf)."""
    if not _A2A_AVAILABLE:
        raise ImportError("a2a-sdk >= 1.1.0 is required.")

    # First turn: embed history so remote agent has context.
    if context_id is None and len(messages) > 1:
        history_lines: list[str] = []
        for msg in messages[:-1]:
            role_label = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_lines.append(f"{role_label}: {_content_to_text(msg.content)}")
        last_text = _content_to_text(messages[-1].content)
        text = (
            f"[Conversation history]\n{chr(10).join(history_lines)}"
            f"\n\n[Current message]\nUser: {last_text}"
        )
    else:
        last_human: Optional[BaseMessage] = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg
                break
        if last_human is None:
            raise A2AProtocolError("No HumanMessage found in message history.")
        text = _content_to_text(last_human.content)

    # Build protobuf Message.
    proto_msg = A2AMessage()
    proto_msg.message_id = str(uuid.uuid4())
    proto_msg.role = A2ARole.Value("ROLE_USER")
    if context_id:
        proto_msg.context_id = context_id
    if task_id:
        proto_msg.task_id = task_id
    part = proto_msg.parts.add()
    part.text = text

    # Wrap in SendMessageRequest.
    req = SendMessageRequest()
    req.message.CopyFrom(proto_msg)
    return req


# ---------------------------------------------------------------------------
# A2A (protobuf) → LangChain
# ---------------------------------------------------------------------------

def _extract_text_from_parts(parts: Any) -> str:
    """Pull plain text out of a repeated protobuf Part field."""
    texts: list[str] = []
    for part in parts:
        content_type = part.WhichOneof("content")
        if content_type == "text":
            if part.text:
                texts.append(part.text)
        elif content_type == "url":
            if part.url:
                texts.append(f"[file: {part.url}]")
        # 'raw' and 'data' are binary/structured – skip for text extraction
    return "\n".join(texts)


def _artifact_to_dict(artifact: "Artifact") -> dict[str, Any]:
    """Serialise a protobuf Artifact to a plain dict."""
    parts_out: list[dict[str, Any]] = []
    for part in artifact.parts:
        content_type = part.WhichOneof("content")
        if content_type == "text":
            parts_out.append({"type": "text", "text": part.text})
        elif content_type == "url":
            entry: dict[str, Any] = {"type": "file", "uri": part.url}
            if part.media_type:
                entry["mime_type"] = part.media_type
            parts_out.append(entry)
        elif content_type == "raw":
            parts_out.append({
                "type": "file",
                "bytes": part.raw.decode("utf-8", errors="replace"),
                "mime_type": part.media_type,
            })
        else:
            parts_out.append({"type": "unknown"})

    return {
        "artifact_id": artifact.artifact_id or None,
        "name": artifact.name or None,
        "parts": parts_out,
        "metadata": None,  # protobuf Struct – omit for simplicity
    }


def a2a_task_to_ai_message(task: "A2ATask") -> AIMessage:
    """Convert a completed protobuf Task into a LangChain AIMessage."""
    text_parts: list[str] = []
    artifacts_out: list[dict[str, Any]] = []

    for artifact in task.artifacts:
        artifact_text = _extract_text_from_parts(artifact.parts)
        if artifact_text:
            text_parts.append(artifact_text)
        artifacts_out.append(_artifact_to_dict(artifact))

    # Fallback: status message
    if not text_parts and task.HasField("status") and task.status.HasField("message"):
        status_text = _extract_text_from_parts(task.status.message.parts)
        if status_text:
            text_parts.append(status_text)

    content = "\n".join(text_parts)
    additional: dict[str, Any] = {
        "a2a_task_id": task.id or None,
        "a2a_context_id": task.context_id or None,
        "a2a_state": task.status.state if task.HasField("status") else None,
        "a2a_artifacts": artifacts_out,
        "a2a_metadata": None,
    }
    return AIMessage(content=content, additional_kwargs=additional)


def a2a_message_to_ai_message(msg: "A2AMessage") -> AIMessage:
    """Convert a direct protobuf Message response into an AIMessage."""
    text = _extract_text_from_parts(msg.parts)
    additional: dict[str, Any] = {
        "a2a_message_id": msg.message_id or None,
        "a2a_context_id": msg.context_id or None,
        "a2a_task_id": msg.task_id or None,
        "a2a_metadata": None,
    }
    return AIMessage(content=text, additional_kwargs=additional)


def status_update_to_text(event: "TaskStatusUpdateEvent") -> Optional[str]:
    """Extract any text payload from a protobuf TaskStatusUpdateEvent."""
    if event.HasField("status") and event.status.HasField("message"):
        return _extract_text_from_parts(event.status.message.parts) or None
    return None


def artifact_update_to_text(event: "TaskArtifactUpdateEvent") -> Optional[str]:
    """Extract any text payload from a protobuf TaskArtifactUpdateEvent."""
    if event.HasField("artifact"):
        return _extract_text_from_parts(event.artifact.parts) or None
    return None
