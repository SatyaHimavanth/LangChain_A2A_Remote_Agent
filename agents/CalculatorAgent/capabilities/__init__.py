"""Capability configuration and runtime component assembly for the agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCapabilities, AgentCard

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentCapabilityConfig:
    """Feature flags and tuning values used by ``build_server_components``."""

    push_notifications: bool = False
    """Post task-state updates to client-registered webhook URLs."""

    task_management: bool = False
    """Use a task store suitable for GetTask/ListTasks/SubscribeToTask flows."""

    multi_modal: bool = False
    """Accept non-text parts and convert them into LangChain content blocks."""

    opaque_execution: bool = False
    """Suppress intermediate WORKING/SUBMITTED events from streaming clients."""

    push_timeout: float = 10.0
    push_retries: int = 3
    push_retry_base_delay: float = 1.0

    task_store_backend: str = "memory"
    task_store_path: str = "./tasks.json"
    task_ttl_seconds: int = 0
    task_ttl_check_interval: int = 60


@dataclass(slots=True)
class ServerComponents:
    """Objects passed into the A2A request handler and app lifespan."""

    task_store: TaskStore
    push_config_store: PushNotificationConfigStore | None
    push_sender: PushNotificationSender | None
    input_modes: list[str] = field(default_factory=lambda: ["text", "text/plain"])
    output_modes: list[str] = field(default_factory=lambda: ["text", "text/plain"])
    _httpx_client: httpx.AsyncClient | None = field(default=None, repr=False)
    _ttl_cleaner: object | None = field(default=None, repr=False)
    _capabilities: AgentCapabilities = field(
        default_factory=lambda: AgentCapabilities(
            streaming=True,
            extended_agent_card=True,
        ),
        repr=False,
    )

    def apply_to_card(self, card: AgentCard) -> None:
        """Copy the runtime capabilities onto an AgentCard before serving it."""
        card.capabilities.CopyFrom(self._capabilities)

    async def startup(self) -> None:
        """Start background capability workers such as TTL cleanup."""
        if self._ttl_cleaner is not None:
            self._ttl_cleaner.start()

    async def shutdown(self) -> None:
        """Stop background workers and close owned network clients."""
        if self._ttl_cleaner is not None:
            await self._ttl_cleaner.stop()
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            logger.info("Push-notification HTTP client closed.")


def build_server_components(config: AgentCapabilityConfig) -> ServerComponents:
    """Build the concrete server dependencies for the enabled capabilities."""
    task_store: TaskStore = InMemoryTaskStore()
    push_config_store: PushNotificationConfigStore | None = None
    push_sender: PushNotificationSender | None = None
    httpx_client: httpx.AsyncClient | None = None
    ttl_cleaner: object | None = None

    capabilities = AgentCapabilities(
        streaming=True,
        extended_agent_card=True,
        push_notifications=False,
    )

    if config.task_management:
        from .task_management import build_task_management_components

        task_store, ttl_cleaner = build_task_management_components(
            backend=config.task_store_backend,
            store_path=config.task_store_path,
            ttl_seconds=config.task_ttl_seconds,
            ttl_check_interval=config.task_ttl_check_interval,
        )

    if config.multi_modal:
        from .multi_modal import MULTIMODAL_INPUT_TYPES, TEXT_OUTPUT_TYPES

        input_modes = list(MULTIMODAL_INPUT_TYPES)
        output_modes = list(TEXT_OUTPUT_TYPES)
        logger.info("Multi-modal input enabled; input modes: %s", input_modes)
    else:
        input_modes = ["text", "text/plain"]
        output_modes = ["text", "text/plain"]

    if config.push_notifications:
        from .push_notifications import build_push_components

        push_config_store, push_sender, httpx_client = build_push_components(
            timeout=config.push_timeout,
            retries=config.push_retries,
            retry_base_delay=config.push_retry_base_delay,
        )
        capabilities.push_notifications = True

    return ServerComponents(
        task_store=task_store,
        push_config_store=push_config_store,
        push_sender=push_sender,
        input_modes=input_modes,
        output_modes=output_modes,
        _httpx_client=httpx_client,
        _ttl_cleaner=ttl_cleaner,
        _capabilities=capabilities,
    )
