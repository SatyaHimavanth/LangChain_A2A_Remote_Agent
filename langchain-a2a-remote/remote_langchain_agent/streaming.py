"""
streaming.py
~~~~~~~~~~~~
Convert a stream of A2A client events into LangChain
:class:`~langchain_core.messages.AIMessageChunk` objects.

The A2A client yields a heterogeneous stream:

* ``tuple[Task, None]``  – initial task created by the server.
* ``tuple[Task, TaskStatusUpdateEvent]`` – state transition + optional text.
* ``tuple[Task, TaskArtifactUpdateEvent]`` – partial artifact delivery.
* ``Message`` – bare message (simple/non-task agents).

:func:`event_to_chunks` maps each of these to zero or more
:class:`AIMessageChunk` instances so the caller can ``yield`` them directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langchain_core.messages import AIMessageChunk

if TYPE_CHECKING:
    from a2a.types import (
        Message as A2AMessage,
        Task as A2ATask,
        TaskArtifactUpdateEvent,
        TaskStatusUpdateEvent,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def event_to_chunks(
    raw_event: Any,
    *,
    include_metadata: bool = True,
) -> list[AIMessageChunk]:
    """Convert one A2A client event into zero or more :class:`AIMessageChunk` objects.

    Args:
        raw_event: A single value yielded by ``a2a.client.Client.send_message``.
            This is either a ``(Task, Optional[Update])`` tuple or a bare
            :class:`~a2a.types.Message`.
        include_metadata: When ``True`` (default), attach A2A event metadata to
            ``AIMessageChunk.additional_kwargs`` for observability.

    Returns:
        A (possibly empty) list of :class:`AIMessageChunk` objects.
    """
    try:
        from a2a.types import (
            Message as A2AMessage,
            TaskArtifactUpdateEvent,
            TaskStatusUpdateEvent,
        )
    except ImportError:  # pragma: no cover
        return []

    chunks: list[AIMessageChunk] = []

    # ------------------------------------------------------------------
    # Case 1: (Task, Optional[Update]) tuple  →  the primary streaming shape.
    # ------------------------------------------------------------------
    if isinstance(raw_event, tuple) and len(raw_event) == 2:
        task, update = raw_event

        if update is None:
            # Initial task-created event: emit an empty chunk with metadata.
            if include_metadata:
                chunks.append(
                    AIMessageChunk(
                        content="",
                        additional_kwargs=_task_metadata(task, event_type="task_created"),
                    )
                )
            return chunks

        if isinstance(update, TaskStatusUpdateEvent):
            chunks.extend(_handle_status_update(update, task, include_metadata))
            return chunks

        if isinstance(update, TaskArtifactUpdateEvent):
            chunks.extend(_handle_artifact_update(update, task, include_metadata))
            return chunks

        return chunks

    # ------------------------------------------------------------------
    # Case 2: bare Message  →  simple / non-task agents.
    # ------------------------------------------------------------------
    if isinstance(raw_event, A2AMessage):
        return _handle_direct_message(raw_event, include_metadata)

    # Unknown shape – return empty rather than raise so the stream is robust.
    return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _task_metadata(task: "A2ATask", *, event_type: str) -> dict[str, Any]:
    return {
        "a2a_event_type": event_type,
        "a2a_task_id": task.id,
        "a2a_context_id": getattr(task, "context_id", None),
        "a2a_state": (
            task.status.state.value if task.status else None
        ),
    }


def _handle_status_update(
    update: "TaskStatusUpdateEvent",
    task: "A2ATask",
    include_metadata: bool,
) -> list[AIMessageChunk]:
    """Emit chunks for a :class:`~a2a.types.TaskStatusUpdateEvent`."""
    from .artifacts import extract_streaming_text

    text = extract_streaming_text(update)
    chunks: list[AIMessageChunk] = []

    meta: dict[str, Any] = {}
    if include_metadata:
        meta = {
            "a2a_event_type": "status_update",
            "a2a_task_id": task.id,
            "a2a_context_id": getattr(task, "context_id", None),
            "a2a_state": update.status.state.value if update.status else None,
            "a2a_final": getattr(update, "final", False),
        }

    if text:
        chunks.append(AIMessageChunk(content=text, additional_kwargs=meta))
    elif include_metadata and getattr(update, "final", False):
        # Final event with no text: emit an empty terminal chunk so callers
        # that watch for the end-of-stream signal still receive it.
        chunks.append(AIMessageChunk(content="", additional_kwargs=meta))

    return chunks


def _handle_artifact_update(
    update: "TaskArtifactUpdateEvent",
    task: "A2ATask",
    include_metadata: bool,
) -> list[AIMessageChunk]:
    """Emit chunks for a :class:`~a2a.types.TaskArtifactUpdateEvent`."""
    from .adapters import _artifact_to_dict
    from .artifacts import extract_streaming_text

    text = extract_streaming_text(update)
    chunks: list[AIMessageChunk] = []

    meta: dict[str, Any] = {}
    if include_metadata:
        meta = {
            "a2a_event_type": "artifact_update",
            "a2a_task_id": task.id,
            "a2a_context_id": getattr(task, "context_id", None),
            "a2a_artifact": _artifact_to_dict(update.artifact),
        }

    if text:
        chunks.append(AIMessageChunk(content=text, additional_kwargs=meta))
    elif include_metadata:
        # Non-text artifact (e.g. file): emit metadata-only chunk.
        chunks.append(AIMessageChunk(content="", additional_kwargs=meta))

    return chunks


def _handle_direct_message(
    msg: "A2AMessage",
    include_metadata: bool,
) -> list[AIMessageChunk]:
    """Emit a chunk for a bare A2A :class:`~a2a.types.Message` response."""
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
    """Merge a list of :class:`AIMessageChunk` objects into a single aggregate chunk.

    Useful for callers that buffer the full stream before returning.

    Args:
        chunks: All chunks from a completed stream.

    Returns:
        A single :class:`AIMessageChunk` with concatenated content and merged
        ``additional_kwargs`` (last non-empty value wins per key).
    """
    if not chunks:
        return AIMessageChunk(content="")

    merged_text = "".join(c.content for c in chunks if c.content)
    merged_meta: dict[str, Any] = {}
    for c in chunks:
        merged_meta.update(c.additional_kwargs)

    return AIMessageChunk(content=merged_text, additional_kwargs=merged_meta)
