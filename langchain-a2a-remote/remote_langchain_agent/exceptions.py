"""
exceptions.py
~~~~~~~~~~~~~
Custom exceptions for remote_langchain_agent.

All exceptions derive from RemoteAgentError so callers can catch the
broad class or any specific subclass as needed.
"""

from __future__ import annotations


class RemoteAgentError(Exception):
    """Base exception for all errors raised by remote_langchain_agent."""


class CardResolutionError(RemoteAgentError):
    """Failed to fetch or parse the remote agent's Agent Card.

    Usually indicates that the URL is wrong, the server is down, or the
    response body is not a valid A2A AgentCard JSON object.
    """


class A2AProtocolError(RemoteAgentError):
    """The remote agent returned a response that violates the A2A protocol.

    Examples:
        - A task with an unexpected final state.
        - A streaming event with a missing required field.
        - An empty response when a message or task was expected.
    """


class A2ATimeoutError(RemoteAgentError):
    """The remote agent did not respond within the configured timeout."""


class A2AAuthError(RemoteAgentError):
    """Authentication or authorisation failure when calling the remote agent.

    Raised when the A2A client receives a 401 or 403 HTTP response.
    """


class A2AStreamError(RemoteAgentError):
    """An error occurred while consuming a streaming response.

    Wraps lower-level network or decode errors that happen mid-stream so
    callers can distinguish them from connection-setup failures.
    """


class InputNormalisationError(RemoteAgentError):
    """The input passed to :class:`RemoteAgent` could not be normalised.

    Raised when the input is not a ``str``, a ``list[BaseMessage]``, or a
    dict with a ``"messages"`` key.
    """
