"""
messages.py
~~~~~~~~~~~
Shared type aliases, message-building utilities, and role-mapping constants.

This module is intentionally thin — it provides the vocabulary that the rest
of the package speaks without coupling any conversion logic to a specific
direction.  Both :mod:`adapters` and :mod:`streaming` import from here.

Role mapping
~~~~~~~~~~~~
The A2A protocol uses two roles: ``"user"`` and ``"agent"``.
LangChain distinguishes ``HumanMessage``, ``AIMessage``, ``SystemMessage``,
and ``ToolMessage``.

Mapping table::

    LangChain          → A2A role
    ─────────────────────────────
    HumanMessage       → "user"
    AIMessage          → "agent"
    SystemMessage      → "user"  (prepended as context; A2A has no system role)
    ToolMessage        → "user"  (tool result returned to agent)

    A2A role           → LangChain
    ────────────────────────────────
    "user"             → HumanMessage  (when building replay history)
    "agent"            → AIMessage
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Role constants (mirrors a2a.types.Role without importing a2a at module level)
# ---------------------------------------------------------------------------

A2A_ROLE_USER: str = "user"
A2A_ROLE_AGENT: str = "agent"

# ---------------------------------------------------------------------------
# LangChain → A2A role
# ---------------------------------------------------------------------------

_LC_TO_A2A: dict[type[BaseMessage], str] = {
    HumanMessage: A2A_ROLE_USER,
    AIMessage: A2A_ROLE_AGENT,
    SystemMessage: A2A_ROLE_USER,   # A2A has no native system role
    ToolMessage: A2A_ROLE_USER,     # tool results feed back as user context
}


def lc_role_to_a2a(message: BaseMessage) -> str:
    """Return the A2A ``role`` string for a LangChain *message*.

    Args:
        message: Any :class:`~langchain_core.messages.BaseMessage`.

    Returns:
        ``"user"`` or ``"agent"``.
    """
    return _LC_TO_A2A.get(type(message), A2A_ROLE_USER)


# ---------------------------------------------------------------------------
# A2A role → LangChain constructor
# ---------------------------------------------------------------------------

_A2A_TO_LC: dict[str, type[BaseMessage]] = {
    A2A_ROLE_USER: HumanMessage,
    A2A_ROLE_AGENT: AIMessage,
}


def a2a_role_to_lc_type(role: str) -> type[BaseMessage]:
    """Return the LangChain message class that corresponds to an A2A *role*.

    Args:
        role: ``"user"`` or ``"agent"`` as returned by the A2A server.

    Returns:
        :class:`~langchain_core.messages.HumanMessage` or
        :class:`~langchain_core.messages.AIMessage`.  Unknown roles fall back
        to :class:`~langchain_core.messages.HumanMessage`.
    """
    return _A2A_TO_LC.get(role, HumanMessage)


# ---------------------------------------------------------------------------
# Type aliases exported for use in type hints across the package
# ---------------------------------------------------------------------------

#: The canonical input type accepted by :class:`RemoteAgent`.
AgentInput = object  # str | HumanMessage | list[BaseMessage] | dict

#: The canonical output type returned by :class:`RemoteAgent`.
AgentOutput = dict  # {"messages": list[BaseMessage], **extra_state}

#: All LangChain message types this package produces.
LCMessage = BaseMessage | AIMessageChunk

__all__ = [
    "A2A_ROLE_USER",
    "A2A_ROLE_AGENT",
    "lc_role_to_a2a",
    "a2a_role_to_lc_type",
    "AgentInput",
    "AgentOutput",
    "LCMessage",
]
