"""Asynchronous push-notification support for A2A task events."""

from __future__ import annotations

import asyncio
import logging

import httpx
from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    PushNotificationConfigStore,
    PushNotificationEvent,
    PushNotificationSender,
)
from a2a.types.a2a_pb2 import TaskPushNotificationConfig
from a2a.utils.proto_utils import to_stream_response
from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)


class RetryablePushNotificationSender(PushNotificationSender):
    """Deliver push notifications with per-endpoint exponential backoff."""

    def __init__(
        self,
        *,
        httpx_client: httpx.AsyncClient,
        config_store: PushNotificationConfigStore,
        max_tries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self._client = httpx_client
        self._config_store = config_store
        self._max_tries = max(1, max_tries)
        self._base_delay = max(0.0, base_delay)

    async def send_notification(
        self,
        task_id: str,
        event: PushNotificationEvent,
    ) -> None:
        """Send one event to every webhook registered for ``task_id``."""
        push_configs = await self._config_store.get_info_for_dispatch(task_id)
        if not push_configs:
            return

        results = await asyncio.gather(
            *(
                self._send_one(task_id=task_id, event=event, push_info=push_info)
                for push_info in push_configs
            )
        )
        if not all(results):
            logger.warning("One or more push notifications failed for task_id=%s.", task_id)

    async def _send_one(
        self,
        *,
        task_id: str,
        event: PushNotificationEvent,
        push_info: TaskPushNotificationConfig,
    ) -> bool:
        """Post a notification to one webhook URL with retries."""
        delay = self._base_delay
        headers = (
            {"X-A2A-Notification-Token": push_info.token}
            if push_info.token
            else None
        )
        payload = MessageToDict(to_stream_response(event))

        for attempt in range(1, self._max_tries + 1):
            try:
                response = await self._client.post(
                    push_info.url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                logger.info(
                    "Push notification sent for task_id=%s to %s.",
                    task_id,
                    push_info.url,
                )
                return True
            except Exception as exc:
                if attempt >= self._max_tries:
                    logger.error(
                        "Push notification failed for task_id=%s to %s after "
                        "%d attempt(s): %s",
                        task_id,
                        push_info.url,
                        self._max_tries,
                        exc,
                    )
                    return False
                logger.warning(
                    "Push notification attempt %d/%d failed for task_id=%s; "
                    "retrying in %.1fs: %s",
                    attempt,
                    self._max_tries,
                    task_id,
                    delay,
                    exc,
                )
                if delay:
                    await asyncio.sleep(delay)
                delay *= 2

        return False


def build_push_components(
    *,
    timeout: float = 10.0,
    retries: int = 3,
    retry_base_delay: float = 1.0,
) -> tuple[PushNotificationConfigStore, PushNotificationSender, httpx.AsyncClient]:
    """Create the config store, sender, and owned HTTP client."""
    config_store = InMemoryPushNotificationConfigStore()
    httpx_client = httpx.AsyncClient(timeout=timeout)
    sender = RetryablePushNotificationSender(
        httpx_client=httpx_client,
        config_store=config_store,
        max_tries=retries,
        base_delay=retry_base_delay,
    )
    logger.info(
        "Push notifications enabled (timeout=%.1fs, retries=%d).",
        timeout,
        retries,
    )
    return config_store, sender, httpx_client
