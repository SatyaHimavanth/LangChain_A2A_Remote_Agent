"""Opaque execution support for hiding intermediate streaming status events."""

from __future__ import annotations

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState, TaskStatusUpdateEvent

logger = logging.getLogger(__name__)

_SUPPRESSED_STATES: frozenset[int] = frozenset(
    {
        TaskState.TASK_STATE_WORKING,
        TaskState.TASK_STATE_SUBMITTED,
    }
)


class OpaqueEventQueue(EventQueue):
    """Forward events to another queue while dropping intermediate statuses."""

    def __init__(self, inner: EventQueue) -> None:
        self._inner = inner

    async def enqueue_event(self, event: object) -> None:
        """Drop WORKING/SUBMITTED status events and forward everything else."""
        if (
            isinstance(event, TaskStatusUpdateEvent)
            and event.status.state in _SUPPRESSED_STATES
        ):
            logger.debug(
                "Opaque mode suppressed %s status event for task_id=%s.",
                TaskState.Name(event.status.state),
                event.task_id,
            )
            return
        await self._inner.enqueue_event(event)  # type: ignore[arg-type]


class OpaqueAgentExecutorWrapper(AgentExecutor):
    """AgentExecutor decorator that applies ``OpaqueEventQueue``."""

    def __init__(self, inner: AgentExecutor) -> None:
        self._inner = inner
        logger.info("Opaque execution enabled.")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        await self._inner.execute(context, OpaqueEventQueue(event_queue))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        await self._inner.cancel(context, OpaqueEventQueue(event_queue))
