"""Background TTL cleaner for task stores that support expiry sweeps."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class CleanableStore(Protocol):
    """Task store protocol required by ``TTLTaskCleaner``."""

    async def cleanup_expired_tasks(self, ttl_seconds: int) -> int: ...


class TTLTaskCleaner:
    """Periodically delete terminal tasks older than a configured TTL."""

    def __init__(
        self,
        store: CleanableStore,
        ttl_seconds: int,
        check_interval_seconds: int = 60,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero.")
        if check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be greater than zero.")
        if not isinstance(store, CleanableStore):
            raise TypeError(f"{type(store).__name__} does not support TTL cleanup.")

        self._store = store
        self._ttl = ttl_seconds
        self._interval = check_interval_seconds
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start the cleanup loop in the current asyncio event loop."""
        if self._task is not None:
            logger.warning("TTLTaskCleaner already running; duplicate start ignored.")
            return
        self._task = asyncio.create_task(self._run(), name="ttl-task-cleaner")
        logger.info(
            "TTLTaskCleaner started (ttl=%ds, interval=%ds).",
            self._ttl,
            self._interval,
        )

    async def stop(self) -> None:
        """Cancel the cleanup loop and wait for shutdown."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
        logger.info("TTLTaskCleaner stopped.")

    async def _run(self) -> None:
        while True:
            await self._sweep()
            await asyncio.sleep(self._interval)

    async def _sweep(self) -> None:
        """Run one cleanup pass, logging failures without killing the loop."""
        try:
            deleted = await self._store.cleanup_expired_tasks(self._ttl)
        except Exception:
            logger.exception("TTL sweep failed; will retry on next interval.")
            return

        if deleted:
            logger.info("TTL sweep removed %d expired task(s).", deleted)
        else:
            logger.debug("TTL sweep found no expired tasks.")
