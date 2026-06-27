"""
adapters.py
~~~~~~~~~~~
Bidirectional conversion between LangChain message types and the A2A
protocol message/part types from ``a2a-sdk ~= 0.3``.

Design notes
~~~~~~~~~~~~
* We only touch ``a2a.types`` here; all transport concerns live in
  :mod:`client`.
* LangChain ``content`` can be a plain ``str`` or a list of content blocks
  (the new ``content_blocks`` format introduced in langchain-core 1.0).  We
  handle both.
* We never discard protocol information: task metadata, artifact metadata,
  and part metadata are preserved in ``AIMessage.additional_kwargs``.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .exceptions import A2AProtocolError, InputNormalisationError

# ---------------------------------------------------------------------------
# A2A imports – guarded so the rest of the codebase can be imported without
# a2a-sdk present (e.g. when only type-checking).
# ---------------------------------------------------------------------------
try:
    from a2a.types import (
        Artifact,
        FilePart,
        Message as A2AMessage,
        Part as A2APart,
        Role as A2ARole,
        Task as A2ATask,
        TaskArtifactUpdateEvent,
        TaskState,
        TaskStatus,
        TaskStatusUpdateEvent,
        TextPart,
    )

    _A2A_AVAILABLE = True
except ImportError:  # pragma: no cover
    _A2A_AVAILABLE = False


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def normalise_input(
    raw: Any,
) -> tuple[list[BaseMessage], dict[str, Any]]:
    """Coerce the many accepted input shapes into a canonical message list.

    Accepted input shapes:

    * ``str`` – wrapped in a :class:`~langchain_core.messages.HumanMessage`.
    * ``HumanMessage`` – returned as-is in a single-element list.
    * ``list[BaseMessage]`` – returned as-is.
    * ``dict`` with ``"messages"`` key – messages extracted; remaining keys
      returned as extra state.

    Args:
        raw: The value passed to :meth:`RemoteAgent.invoke` / ``ainvoke``.

    Returns:
        A ``(messages, extra_state)`` tuple where *extra_state* is any dict
        payload that was in the original input besides ``"messages"``.

    Raises:
        :class:`~remote_langchain_agent.exceptions.InputNormalisationError`:
            When the input cannot be interpreted.
    """
    extra: dict[str, Any] = {}

    if isinstance(raw, str):
        return [HumanMessage(content=raw)], extra

    if isinstance(raw, HumanMessage):
        return [raw], extra

    if isinstance(raw, list):
        if not raw:
            raise InputNormalisationError("Input list is empty; expected at least one message.")
        if not all(isinstance(m, BaseMessage) for m in raw):
            raise InputNormalisationError(
                "Input list must contain only LangChain BaseMessage instances."
            )
        return raw, extra

    if isinstance(raw, dict):
        messages_raw = raw.get("messages")
        if messages_raw is None:
            raise InputNormalisationError(
                'Input dict must have a "messages" key containing a list of BaseMessage.'
            )
        # Accept pre-parsed BaseMessages or raw dicts (role/content pairs).
        messages: list[BaseMessage] = []
        for item in messages_raw:
            if isinstance(item, BaseMessage):
                messages.append(item)
            elif isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
            else:
                raise InputNormalisationError(
                    f"Unexpected item in messages list: {type(item)}"
                )
        extra = {k: v for k, v in raw.items() if k != "messages"}
        return messages, extra

    raise InputNormalisationError(
        f"Cannot normalise input of type {type(raw).__name__}. "
        "Expected str, HumanMessage, list[BaseMessage], or dict with 'messages' key."
    )


# ---------------------------------------------------------------------------
# LangChain → A2A
# ---------------------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    """Extract plain text from a LangChain message content value.

    Handles both the legacy ``str`` form and the modern content-block list.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                # Non-text blocks (images, tool_use, etc.) are skipped; the
                # remote agent receives a text-only view.
        return "\n".join(parts)
    return str(content)


def lc_messages_to_a2a_message(
    messages: list[BaseMessage],
    *,
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> "A2AMessage":
    """Convert a LangChain message list into a single A2A :class:`Message`.

    Only the **last** human (user) message in *messages* is sent as the
    current turn content.  Earlier messages are not re-sent because the A2A
    server tracks conversation history via ``context_id``.

    For the very first turn (``context_id`` is ``None``) the complete history
    is collapsed and sent as a single formatted text block so the remote agent
    has enough context to reason.

    Args:
        messages: The full conversation history from LangChain state.
        context_id: If set, included in the A2A message so the server can
            resume the existing session.
        task_id: If set, included to attach this message to an existing task.

    Returns:
        An :class:`a2a.types.Message` ready to be sent via the A2A client.

    Raises:
        :class:`~remote_langchain_agent.exceptions.A2AProtocolError`:
            When no human message is found in *messages*.
    """
    if not _A2A_AVAILABLE:
        raise ImportError("a2a-sdk is required.  Install with: pip install 'a2a-sdk~=0.3'")

    # ------------------------------------------------------------------
    # First turn: embed history context so the remote agent is not blind.
    # Subsequent turns: send only the new user message.
    # ------------------------------------------------------------------
    if context_id is None and len(messages) > 1:
        # Build a history preamble from all messages before the last one.
        history_lines: list[str] = []
        for msg in messages[:-1]:
            role_label = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_lines.append(f"{role_label}: {_content_to_text(msg.content)}")
        history_text = "\n".join(history_lines)

        last_msg = messages[-1]
        last_text = _content_to_text(last_msg.content)
        combined = f"[Conversation history]\n{history_text}\n\n[Current message]\nUser: {last_text}"
        parts = [A2APart(root=TextPart(text=combined))]
    else:
        # Find the last HumanMessage (the actual new query this turn).
        last_human: Optional[BaseMessage] = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg
                break

        if last_human is None:
            raise A2AProtocolError(
                "No HumanMessage found in message history. "
                "The last message must be from the user."
            )

        text = _content_to_text(last_human.content)
        parts = [A2APart(root=TextPart(text=text))]

    return A2AMessage(
        message_id=str(uuid.uuid4()),
        role=A2ARole.user,
        parts=parts,
        context_id=context_id,
        task_id=task_id,
    )


# ---------------------------------------------------------------------------
# A2A → LangChain
# ---------------------------------------------------------------------------


def _extract_text_from_parts(parts: list["A2APart"]) -> str:
    """Pull plain text out of an A2A parts list, joining with newlines."""
    texts: list[str] = []
    for part in parts:
        root = part.root
        if isinstance(root, TextPart):
            texts.append(root.text)
        elif isinstance(root, FilePart):
            # Include the URI or MIME type as a note so text consumers see
            # something useful.
            if hasattr(root.file, "uri") and root.file.uri:
                texts.append(f"[file: {root.file.uri}]")
            elif hasattr(root.file, "mime_type") and root.file.mime_type:
                texts.append(f"[file: {root.file.mime_type}]")
        # DataPart is intentionally not converted to text here; callers that
        # need structured data should inspect additional_kwargs["artifacts"].
    return "\n".join(texts)


def _artifact_to_dict(artifact: "Artifact") -> dict[str, Any]:
    """Serialise an A2A Artifact to a plain dict for storage in
    ``AIMessage.additional_kwargs``."""
    parts_out: list[dict[str, Any]] = []
    for part in artifact.parts:
        root = part.root
        if isinstance(root, TextPart):
            parts_out.append({"type": "text", "text": root.text})
        elif isinstance(root, FilePart):
            file_dict: dict[str, Any] = {"type": "file"}
            if hasattr(root, "file"):
                f = root.file
                if hasattr(f, "uri") and f.uri:
                    file_dict["uri"] = f.uri
                if hasattr(f, "mime_type") and f.mime_type:
                    file_dict["mime_type"] = f.mime_type
                if hasattr(f, "bytes") and f.bytes:
                    file_dict["bytes"] = f.bytes  # base64 str
            parts_out.append(file_dict)
        else:
            parts_out.append({"type": "unknown"})

    return {
        "artifact_id": getattr(artifact, "artifact_id", None),
        "name": getattr(artifact, "name", None),
        "parts": parts_out,
        "metadata": getattr(artifact, "metadata", None),
    }


def a2a_task_to_ai_message(task: "A2ATask") -> AIMessage:
    """Convert a completed A2A :class:`Task` into a LangChain :class:`AIMessage`.

    Text is assembled from ``task.artifacts`` first; if no text is found
    there, the status message (if any) is used as a fallback.

    All A2A metadata, artifact details, and task status are preserved in
    ``AIMessage.additional_kwargs`` under the keys ``"a2a_task_id"``,
    ``"a2a_context_id"``, ``"a2a_state"``, ``"a2a_artifacts"``, and
    ``"a2a_metadata"``.

    Args:
        task: The :class:`a2a.types.Task` returned by the remote agent.

    Returns:
        :class:`~langchain_core.messages.AIMessage` carrying the agent's reply.
    """
    text_parts: list[str] = []
    artifacts_out: list[dict[str, Any]] = []

    # Primary source: task artifacts.
    if task.artifacts:
        for artifact in task.artifacts:
            artifact_text = _extract_text_from_parts(artifact.parts)
            if artifact_text:
                text_parts.append(artifact_text)
            artifacts_out.append(_artifact_to_dict(artifact))

    # Fallback: status message (common when agent replies with a Message, not a Task).
    if not text_parts and task.status and task.status.message:
        status_text = _extract_text_from_parts(task.status.message.parts)
        if status_text:
            text_parts.append(status_text)

    content = "\n".join(text_parts)

    additional: dict[str, Any] = {
        "a2a_task_id": task.id,
        "a2a_context_id": task.context_id,
        "a2a_state": task.status.state.value if task.status else None,
        "a2a_artifacts": artifacts_out,
        "a2a_metadata": getattr(task, "metadata", None),
    }

    return AIMessage(content=content, additional_kwargs=additional)


def a2a_message_to_ai_message(msg: "A2AMessage") -> AIMessage:
    """Convert a direct A2A :class:`Message` response into an :class:`AIMessage`.

    Some A2A servers (particularly simple ``helloworld``-style agents) respond
    with a bare :class:`~a2a.types.Message` rather than wrapping everything in
    a :class:`~a2a.types.Task`.  This converter handles that case.

    Args:
        msg: The :class:`a2a.types.Message` sent by the remote agent.

    Returns:
        :class:`~langchain_core.messages.AIMessage`.
    """
    text = _extract_text_from_parts(msg.parts)
    additional: dict[str, Any] = {
        "a2a_message_id": msg.message_id,
        "a2a_context_id": getattr(msg, "context_id", None),
        "a2a_task_id": getattr(msg, "task_id", None),
        "a2a_metadata": getattr(msg, "metadata", None),
    }
    return AIMessage(content=text, additional_kwargs=additional)


def status_update_to_text(event: "TaskStatusUpdateEvent") -> Optional[str]:
    """Extract any text payload from a :class:`~a2a.types.TaskStatusUpdateEvent`.

    Returns ``None`` when the event carries no text (e.g. a pure state-change
    notification with no message parts).
    """
    if event.status and event.status.message:
        text = _extract_text_from_parts(event.status.message.parts)
        return text or None
    return None


def artifact_update_to_text(event: "TaskArtifactUpdateEvent") -> Optional[str]:
    """Extract any text payload from a :class:`~a2a.types.TaskArtifactUpdateEvent`."""
    return _extract_text_from_parts(event.artifact.parts) or None
