"""
streaming.py
~~~~~~~~~~~~
Convert a stream of A2A client events into LangChain AIMessageChunk objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langchain_core.messages import AIMessageChunk

if TYPE_CHECKING:
    from a2a.types import (
        Message as A2AMessage,
        StreamResponse,
        Task as A2ATask,
        TaskArtifactUpdateEvent,
        TaskStatusUpdateEvent,
    )


def _import_streaming_types():
    """Return current A2A streaming types."""
    try:
        from a2a.types import (
            Message as A2AMessage,
            StreamResponse,
            TaskArtifactUpdateEvent,
            TaskStatusUpdateEvent,
        )
        return A2AMessage, StreamResponse, TaskStatusUpdateEvent, TaskArtifactUpdateEvent
    except ImportError:
        from a2a.compat.v0_3.types import (
            Message as A2AMessage,
            StreamResponse,
            TaskArtifactUpdateEvent,
            TaskStatusUpdateEvent,
        )
        return A2AMessage, StreamResponse, TaskStatusUpdateEvent, TaskArtifactUpdateEvent


def event_to_chunks(
    raw_event: Any,
    *,
    include_metadata: bool = True,
) -> list[AIMessageChunk]:
    """Convert one A2A client event into zero or more AIMessageChunk objects."""
    try:
        (
            A2AMessage,
            StreamResponse,
            TaskStatusUpdateEvent,
            TaskArtifactUpdateEvent,
        ) = _import_streaming_types()
    except ImportError:
        return []

    chunks: list[AIMessageChunk] = []

    if isinstance(raw_event, StreamResponse):
        payload_type = raw_event.WhichOneof("payload")
        if payload_type == "task":
            if include_metadata:
                chunks.append(
                    AIMessageChunk(
                        content="",
                        additional_kwargs=_task_metadata(
                            raw_event.task,
                            event_type="task",
                        ),
                    )
                )
            return chunks
        if payload_type == "message":
            return _handle_direct_message(raw_event.message, include_metadata)
        if payload_type == "status_update":
            chunks.extend(_handle_status_update(raw_event.status_update, include_metadata))
            return chunks
        if payload_type == "artifact_update":
            chunks.extend(_handle_artifact_update(raw_event.artifact_update, include_metadata))
            return chunks
        return chunks

    if isinstance(raw_event, tuple) and len(raw_event) == 2:
        task, update = raw_event

        if update is None:
            if include_metadata:
                chunks.append(
                    AIMessageChunk(
                        content="",
                        additional_kwargs=_task_metadata(task, event_type="task_created"),
                    )
                )
            return chunks

        if isinstance(update, TaskStatusUpdateEvent):
            chunks.extend(_handle_status_update(update, include_metadata, task=task))
            return chunks

        if isinstance(update, TaskArtifactUpdateEvent):
            chunks.extend(_handle_artifact_update(update, include_metadata, task=task))
            return chunks

        return chunks

    if isinstance(raw_event, A2AMessage):
        return _handle_direct_message(raw_event, include_metadata)

    return []


def _task_metadata(task: "A2ATask", *, event_type: str) -> dict[str, Any]:
    return {
        "a2a_event_type": event_type,
        "a2a_task_id": task.id,
        "a2a_context_id": getattr(task, "context_id", None),
        "a2a_state": task.status.state if task.HasField("status") else None,
    }


def _handle_status_update(
    update: "TaskStatusUpdateEvent",
    include_metadata: bool,
    task: Optional["A2ATask"] = None,
) -> list[AIMessageChunk]:
    from .artifacts import extract_streaming_text

    text = extract_streaming_text(update)
    chunks: list[AIMessageChunk] = []
    state_name = _status_state_name(update)

    meta: dict[str, Any] = {}
    if include_metadata:
        meta = {
            "a2a_event_type": "status_update",
            "a2a_task_id": update.task_id or getattr(task, "id", None),
            "a2a_context_id": update.context_id or getattr(task, "context_id", None),
            "a2a_state": update.status.state if update.HasField("status") else None,
        }

    if text:
        if state_name == "TASK_STATE_WORKING":
            chunks.append(
                AIMessageChunk(
                    content="",
                    additional_kwargs={
                        **meta,
                        "reasoning_content": text,
                        "a2a_reasoning": text,
                    },
                )
            )
        else:
            chunks.append(AIMessageChunk(content=text, additional_kwargs=meta))

    return chunks


def _handle_artifact_update(
    update: "TaskArtifactUpdateEvent",
    include_metadata: bool,
    task: Optional["A2ATask"] = None,
) -> list[AIMessageChunk]:
    from .adapters import _artifact_to_dict
    from .artifacts import extract_streaming_text

    text = extract_streaming_text(update)
    chunks: list[AIMessageChunk] = []

    meta: dict[str, Any] = {}
    if include_metadata:
        meta = {
            "a2a_event_type": "artifact_update",
            "a2a_task_id": update.task_id or getattr(task, "id", None),
            "a2a_context_id": update.context_id or getattr(task, "context_id", None),
            "a2a_artifact": _artifact_to_dict(update.artifact),
            "a2a_last_chunk": update.last_chunk,
        }

    if text:
        chunks.append(AIMessageChunk(content=text, additional_kwargs=meta))
    elif include_metadata:
        chunks.append(AIMessageChunk(content="", additional_kwargs=meta))

    return chunks


def _handle_direct_message(
    msg: "A2AMessage",
    include_metadata: bool,
) -> list[AIMessageChunk]:
    from .adapters import _extract_text_from_parts

    text = _extract_text_from_parts(msg.parts)
    meta: dict[str, Any] = {}
    if include_metadata:
        meta = {
            "a2a_event_type": "message",
            "a2a_message_id": msg.message_id,
            "a2a_context_id": getattr(msg, "context_id", None),
        }
    return [AIMessageChunk(content=text, additional_kwargs=meta)]


def chunks_to_final_message(chunks: list[AIMessageChunk]) -> AIMessageChunk:
    """Merge a list of AIMessageChunk objects into a single aggregate chunk."""
    if not chunks:
        return AIMessageChunk(content="")

    merged_text = "".join(c.content for c in chunks if c.content)
    merged_meta: dict[str, Any] = {}
    for c in chunks:
        merged_meta.update(c.additional_kwargs)

    return AIMessageChunk(content=merged_text, additional_kwargs=merged_meta)


def _status_state_name(update: "TaskStatusUpdateEvent") -> str:
    if not update.HasField("status"):
        return ""
    try:
        from a2a.types import TaskState

        return TaskState.Name(update.status.state)
    except Exception:
        return str(update.status.state)
