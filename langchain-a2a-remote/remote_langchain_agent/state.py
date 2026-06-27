"""
state.py
~~~~~~~~
Per-conversation state tracking for multi-turn A2A sessions.

The A2A protocol links turns in a conversation via a ``context_id`` returned
by the server on the first message.  ``ThreadState`` stores that ID (and the
most recent ``task_id``) keyed by the LangChain ``thread_id`` config value.

``ConversationStateStore`` is the thread-safe container used by
:class:`RemoteAgent <remote_langchain_agent.agent.RemoteAgent>`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # Avoid importing a2a at module level so that the package can be imported
    # even if a2a-sdk is not yet installed (e.g. during type-checking only).
    from a2a.types import Task as A2ATask


@dataclass
class ThreadState:
    """Mutable state for a single conversation thread.

    Attributes:
        context_id: The A2A ``context_id`` returned by the server.  Sent back
            on every subsequent turn so the server can resume the session.
        task_id: The most recent A2A ``task_id``.  Kept for diagnostics and
            optional cancellation.
        created_at: UNIX timestamp of first use.
        last_used: UNIX timestamp of most recent use; updated on every call.
    """

    context_id: Optional[str] = None
    task_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def touch(self) -> None:
        """Update ``last_used`` to now."""
        self.last_used = time.time()

    def update_from_task(self, task: "A2ATask") -> None:
        """Persist identifiers from an A2A Task into this state.

        Called after each successful call so the next turn can resume the
        same server-side session.

        Args:
            task: The :class:`a2a.types.Task` returned by the remote agent.
        """
        self.task_id = task.id
        if task.context_id:
            self.context_id = task.context_id
        self.touch()

    def reset(self) -> None:
        """Clear conversation identifiers to start a fresh session."""
        self.context_id = None
        self.task_id = None
        self.touch()

    # --------------------------------------------------------------------------
    # Representation helpers
    # --------------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of this state."""
        return {
            "context_id": self.context_id,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "last_used": self.last_used,
        }


class ConversationStateStore:
    """Thread-safe store that maps *thread_id* → :class:`ThreadState`.

    One store is created per :class:`RemoteAgent
    <remote_langchain_agent.agent.RemoteAgent>` instance.  Thread IDs are
    taken from ``RunnableConfig["configurable"]["thread_id"]``; when none is
    supplied the agent uses the reserved key ``"__default__"``.
    """

    DEFAULT_THREAD_ID: str = "__default__"

    def __init__(self) -> None:
        self._states: dict[str, ThreadState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, thread_id: Optional[str] = None) -> ThreadState:
        """Return the :class:`ThreadState` for *thread_id*, creating one if absent.

        Args:
            thread_id: Identifier from ``RunnableConfig["configurable"]["thread_id"]``.
                Defaults to :attr:`DEFAULT_THREAD_ID` when ``None``.

        Returns:
            The existing or newly created :class:`ThreadState`.
        """
        key = thread_id or self.DEFAULT_THREAD_ID
        with self._lock:
            if key not in self._states:
                self._states[key] = ThreadState()
            return self._states[key]

    def reset(self, thread_id: Optional[str] = None) -> None:
        """Clear conversation context for *thread_id* without removing the record.

        Args:
            thread_id: Thread to reset.  Defaults to :attr:`DEFAULT_THREAD_ID`.
        """
        self.get(thread_id).reset()

    def delete(self, thread_id: Optional[str] = None) -> None:
        """Remove the state record for *thread_id* entirely.

        Args:
            thread_id: Thread to delete.  Defaults to :attr:`DEFAULT_THREAD_ID`.
        """
        key = thread_id or self.DEFAULT_THREAD_ID
        with self._lock:
            self._states.pop(key, None)

    def clear_all(self) -> None:
        """Remove all stored state (useful in tests)."""
        with self._lock:
            self._states.clear()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a read-only snapshot of all thread states.

        Returns:
            Mapping of thread_id → serialisable dict.
        """
        with self._lock:
            return {k: v.as_dict() for k, v in self._states.items()}
