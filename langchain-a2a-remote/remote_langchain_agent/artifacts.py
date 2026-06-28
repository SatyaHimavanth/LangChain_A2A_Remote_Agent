"""
artifacts.py
~~~~~~~~~~~~
Higher-level helpers for extracting structured content from A2A Task
artifacts and event payloads.
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


def _import_event_types():
    """Return current A2A event types."""
    try:
        from a2a.types import (
            TaskArtifactUpdateEvent,
            TaskState,
            TaskStatusUpdateEvent,
        )
        return TaskStatusUpdateEvent, TaskArtifactUpdateEvent, TaskState
    except ImportError:
        from a2a.compat.v0_3.types import (
            TaskArtifactUpdateEvent,
            TaskState,
            TaskStatusUpdateEvent,
        )
        return TaskStatusUpdateEvent, TaskArtifactUpdateEvent, TaskState


def extract_final_text(task: "A2ATask") -> str:
    """Return the complete text output of a finished A2A task."""
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
    """Pull text from a streaming A2A update event."""
    from .adapters import artifact_update_to_text, status_update_to_text

    try:
        TaskStatusUpdateEvent, TaskArtifactUpdateEvent, _ = _import_event_types()
    except ImportError:
        return None

    if isinstance(event, TaskStatusUpdateEvent):
        return status_update_to_text(event)
    if isinstance(event, TaskArtifactUpdateEvent):
        return artifact_update_to_text(event)
    return None


def is_terminal_state(task: "A2ATask") -> bool:
    """Return True when the task has reached a terminal A2A state."""
    try:
        _, _, TaskState = _import_event_types()
    except ImportError:
        return False

    if not task.status:
        return False
    return task.status.state in (
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    )


def collect_all_artifacts(task: "A2ATask") -> list[dict[str, Any]]:
    """Return all task artifacts as serialisable dicts."""
    from .adapters import _artifact_to_dict

    if not task.artifacts:
        return []
    return [_artifact_to_dict(a) for a in task.artifacts]
