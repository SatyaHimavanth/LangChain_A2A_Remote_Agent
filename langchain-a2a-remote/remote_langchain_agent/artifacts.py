"""
artifacts.py
~~~~~~~~~~~~
Higher-level helpers for extracting structured content from A2A Task
artifacts and event payloads.

These are used by both the synchronous result path and the streaming path so
that callers always get a consistent representation regardless of how the
remote agent communicated its output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from a2a.types import (
        Artifact,
        Task as A2ATask,
        TaskArtifactUpdateEvent,
        TaskStatusUpdateEvent,
    )


def extract_final_text(task: "A2ATask") -> str:
    """Return the complete text output of a finished A2A task.

    Precedence:
    1. Text parts in ``task.artifacts`` (in artifact order).
    2. Text parts in ``task.status.message`` (fallback for simple agents).
    3. Empty string when neither is present.

    Args:
        task: A completed or failed :class:`~a2a.types.Task`.

    Returns:
        Concatenated text content, joined by newlines between artifacts.
    """
    from .adapters import _extract_text_from_parts

    sections: list[str] = []

    if task.artifacts:
        for artifact in task.artifacts:
            text = _extract_text_from_parts(artifact.parts)
            if text:
                sections.append(text)

    if not sections and task.status and task.status.message:
        text = _extract_text_from_parts(task.status.message.parts)
        if text:
            sections.append(text)

    return "\n".join(sections)


def extract_streaming_text(
    event: "TaskStatusUpdateEvent | TaskArtifactUpdateEvent",
) -> Optional[str]:
    """Pull text from a streaming A2A update event.

    Works for both :class:`~a2a.types.TaskStatusUpdateEvent` and
    :class:`~a2a.types.TaskArtifactUpdateEvent`.

    Args:
        event: A streaming event yielded by the A2A client.

    Returns:
        Text string, or ``None`` when the event carries no text payload.
    """
    from .adapters import artifact_update_to_text, status_update_to_text

    try:
        from a2a.types import TaskArtifactUpdateEvent, TaskStatusUpdateEvent
    except ImportError:  # pragma: no cover
        return None

    if isinstance(event, TaskStatusUpdateEvent):
        return status_update_to_text(event)
    if isinstance(event, TaskArtifactUpdateEvent):
        return artifact_update_to_text(event)
    return None


def is_terminal_state(task: "A2ATask") -> bool:
    """Return ``True`` when the task has reached a terminal A2A state.

    Terminal states are: ``completed``, ``failed``, ``canceled``.

    Args:
        task: The :class:`~a2a.types.Task` to inspect.
    """
    try:
        from a2a.types import TaskState
    except ImportError:  # pragma: no cover
        return False

    if not task.status:
        return False
    return task.status.state in (
        TaskState.completed,
        TaskState.failed,
        TaskState.canceled,
    )


def collect_all_artifacts(task: "A2ATask") -> list[dict[str, Any]]:
    """Return all task artifacts as serialisable dicts.

    Useful for callers that want to inspect non-text outputs (file references,
    structured data, etc.) after a call completes.

    Args:
        task: A :class:`~a2a.types.Task` (may be in any state).

    Returns:
        List of artifact dicts; empty list when ``task.artifacts`` is ``None``.
    """
    from .adapters import _artifact_to_dict

    if not task.artifacts:
        return []
    return [_artifact_to_dict(a) for a in task.artifacts]
