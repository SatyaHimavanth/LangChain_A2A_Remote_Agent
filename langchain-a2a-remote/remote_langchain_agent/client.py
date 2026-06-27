"""
client.py
~~~~~~~~~
Manages the lifecycle of an ``a2a-sdk`` client for a single remote A2A agent.

Responsibilities
~~~~~~~~~~~~~~~~
* Lazily fetch and cache the remote agent's Agent Card.
* Construct an ``a2a.client.Client`` from the card via ``ClientFactory``.
* Provide a thin ``send_message`` wrapper with timeout and retry logic.
* Expose the cached :class:`~a2a.types.AgentCard` for capability inspection.

Thread / async safety
~~~~~~~~~~~~~~~~~~~~~
:class:`A2AClientManager` is designed to be created once per
:class:`~remote_langchain_agent.agent.RemoteAgent` instance and then called
from many async tasks concurrently.  Initialisation is protected by an
``asyncio.Lock`` so the card is only fetched once.

Compatibility
~~~~~~~~~~~~~
Targets ``a2a-sdk ~= 0.3.22`` which is pinned by ``google-adk[a2a]``.
The ``a2a-sdk`` 1.x compatibility-mode API is also supported because the
type names and client interface are stable across that boundary.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlparse

import httpx

from .exceptions import (
    A2AAuthError,
    A2AProtocolError,
    A2AStreamError,
    A2ATimeoutError,
    CardResolutionError,
)

logger = logging.getLogger(__name__)

# Sentinel so we can distinguish "not yet initialised" from None.
_UNSET: Any = object()


class A2AClientManager:
    """Owns the A2A SDK client for one remote agent endpoint.

    Args:
        agent_card_url:
            Full URL to the agent card JSON, e.g.
            ``http://localhost:8001/.well-known/agent.json``.
        headers:
            HTTP headers to attach to every request (auth tokens, API keys,
            custom tracing headers, …).
        timeout:
            Total request timeout in seconds applied to card resolution and
            each ``send_message`` call.
        use_client_preference:
            When ``True`` (default) the ``ClientFactory`` is allowed to
            negotiate the transport protocol from the server's capabilities.
        supported_transports:
            List of ``a2a.types.TransportProtocol`` values that the client
            will advertise.  Defaults to both ``http_json`` and ``jsonrpc``.
        httpx_client:
            Provide a pre-configured :class:`httpx.AsyncClient` if you need
            custom TLS, auth middleware, or connection limits.  The manager
            will **not** close a client it did not create.
    """

    def __init__(
        self,
        agent_card_url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 60.0,
        streaming: bool = True,
        use_client_preference: bool = True,
        supported_transports: Optional[list[Any]] = None,
        httpx_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._agent_card_url = agent_card_url
        self._headers = headers or {}
        self._timeout = timeout
        self._streaming = streaming
        self._use_client_preference = use_client_preference
        self._supported_transports_raw = supported_transports  # resolved lazily
        self._external_httpx_client = httpx_client
        self._owns_httpx_client = httpx_client is None

        # State populated on first call to _ensure_initialised().
        self._httpx_client: Optional[httpx.AsyncClient] = httpx_client
        self._agent_card: Any = _UNSET  # a2a.types.AgentCard once resolved
        self._a2a_client: Any = _UNSET  # a2a.client.Client once created
        self._init_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def agent_card(self) -> Optional[Any]:
        """The cached :class:`~a2a.types.AgentCard`, or ``None`` before init."""
        return None if self._agent_card is _UNSET else self._agent_card

    @property
    def is_initialised(self) -> bool:
        """``True`` once the agent card has been fetched and the client built."""
        return self._a2a_client is not _UNSET

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def ensure_initialised(self) -> None:
        """Fetch the Agent Card and build the A2A client (idempotent).

        Raises:
            :class:`~remote_langchain_agent.exceptions.CardResolutionError`:
                When the card cannot be fetched or parsed.
            :class:`~remote_langchain_agent.exceptions.A2AAuthError`:
                When the card endpoint returns 401 / 403.
        """
        if self.is_initialised:
            return
        async with self._init_lock:
            # Double-checked locking: another coroutine may have finished
            # while we were waiting for the lock.
            if self.is_initialised:
                return
            await self._do_init()

    async def _do_init(self) -> None:
        """Perform the actual initialisation (called exactly once)."""
        try:
            from a2a.client import ClientFactory
            from a2a.client.card_resolver import A2ACardResolver
            from a2a.client.client import ClientConfig
            from a2a.types import TransportProtocol
        except ImportError as exc:
            raise ImportError(
                "a2a-sdk is required.  Install with: pip install 'a2a-sdk~=0.3'"
            ) from exc

        # Build (or reuse) the httpx client.
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(
                headers=self._headers,
                timeout=httpx.Timeout(self._timeout),
            )

        # Resolve the card URL into base URL + relative path.
        parsed = urlparse(self._agent_card_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        # The card resolver expects the path relative to the base.
        relative_path = parsed.path or "/.well-known/agent.json"

        logger.debug("Resolving agent card from %s", self._agent_card_url)
        try:
            resolver = A2ACardResolver(
                httpx_client=self._httpx_client,
                base_url=base_url,
            )
            # a2a-sdk 0.3.x exposes get_agent_card() both with and without the
            # relative_card_path kwarg depending on the minor version.
            try:
                card = await resolver.get_agent_card(
                    relative_card_path=relative_path
                )
            except TypeError:
                # Older patch version that takes no argument (uses the
                # /.well-known/agent.json default).
                card = await resolver.get_agent_card()

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise A2AAuthError(
                    f"Authentication failed resolving agent card from "
                    f"{self._agent_card_url}: HTTP {exc.response.status_code}"
                ) from exc
            raise CardResolutionError(
                f"HTTP error {exc.response.status_code} resolving agent card "
                f"from {self._agent_card_url}"
            ) from exc
        except Exception as exc:
            raise CardResolutionError(
                f"Failed to resolve agent card from {self._agent_card_url}: {exc}"
            ) from exc

        self._agent_card = card
        logger.debug("Agent card resolved: %s", getattr(card, "name", "unknown"))

        # Build the a2a client.
        transports = self._supported_transports_raw
        if transports is None:
            transports = [TransportProtocol.http_json, TransportProtocol.jsonrpc]

        config = ClientConfig(
            httpx_client=self._httpx_client,
            streaming=self._streaming,
            supported_transports=transports,
            use_client_preference=self._use_client_preference,
        )
        factory = ClientFactory(config=config)
        self._a2a_client = factory.create(card)
        logger.debug("A2A client created for %s", self._agent_card_url)

    async def close(self) -> None:
        """Release resources owned by this manager.

        Only closes the :class:`httpx.AsyncClient` if it was created
        internally (i.e. not supplied by the caller).
        """
        if self._owns_httpx_client and self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None

    # ------------------------------------------------------------------
    # Message sending
    # ------------------------------------------------------------------

    async def send_message(
        self,
        message: Any,  # a2a.types.Message
        *,
        request_metadata: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        """Send an A2A message and yield raw client events.

        This is a thin async generator that wraps the underlying
        ``a2a.client.Client.send_message`` with timeout and auth-error
        detection.

        Args:
            message: An :class:`~a2a.types.Message` to send.
            request_metadata: Optional metadata dict passed through to the A2A
                client (e.g. session tokens, tracing context).

        Yields:
            Raw events from the A2A client:
            ``tuple[Task, Optional[TaskStatusUpdateEvent | TaskArtifactUpdateEvent]]``
            or a bare :class:`~a2a.types.Message`.

        Raises:
            :class:`~remote_langchain_agent.exceptions.A2ATimeoutError`:
                When the request exceeds ``self._timeout``.
            :class:`~remote_langchain_agent.exceptions.A2AAuthError`:
                On 401 / 403 responses mid-stream.
            :class:`~remote_langchain_agent.exceptions.A2AStreamError`:
                On other network or decode errors during streaming.
        """
        await self.ensure_initialised()

        send_kwargs: dict[str, Any] = {"request": message}
        if request_metadata is not None:
            send_kwargs["request_metadata"] = request_metadata

        try:
            async with asyncio.timeout(self._timeout):
                async for event in self._a2a_client.send_message(**send_kwargs):
                    yield event
        except TimeoutError as exc:
            raise A2ATimeoutError(
                f"Remote agent at {self._agent_card_url} did not respond "
                f"within {self._timeout}s."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise A2AAuthError(
                    f"Authentication error calling remote agent "
                    f"({exc.response.status_code})."
                ) from exc
            raise A2AStreamError(
                f"HTTP error {exc.response.status_code} during streaming."
            ) from exc
        except (httpx.RequestError, httpx.StreamError) as exc:
            raise A2AStreamError(
                f"Network error during streaming from {self._agent_card_url}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "A2AClientManager":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
