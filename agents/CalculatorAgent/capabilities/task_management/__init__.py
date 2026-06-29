"""Factory for task-store and TTL cleanup capability wiring."""

from __future__ import annotations

import logging

from a2a.server.tasks import InMemoryTaskStore, TaskStore

from .ttl_cleanup import TTLTaskCleaner

logger = logging.getLogger(__name__)


def build_task_management_components(
    *,
    backend: str = "memory",
    store_path: str = "./tasks.json",
    ttl_seconds: int = 0,
    ttl_check_interval: int = 60,
) -> tuple[TaskStore, TTLTaskCleaner | None]:
    """Return the configured task store and optional background TTL cleaner."""
    normalized_backend = backend.strip().lower()

    if normalized_backend == "file":
        from .file_store import FileBackedTaskStore

        store: TaskStore = FileBackedTaskStore(path=store_path)
        logger.info("Task management: using FileBackedTaskStore at %s", store_path)
    elif normalized_backend == "memory":
        store = InMemoryTaskStore()
        logger.info("Task management: using InMemoryTaskStore.")
    else:
        raise ValueError(
            "Unsupported task_store_backend "
            f"{backend!r}. Expected 'memory' or 'file'."
        )

    cleaner: TTLTaskCleaner | None = None
    if ttl_seconds > 0:
        from .ttl_cleanup import CleanableStore

        if isinstance(store, CleanableStore):
            cleaner = TTLTaskCleaner(
                store=store,  # type: ignore[arg-type]
                ttl_seconds=ttl_seconds,
                check_interval_seconds=ttl_check_interval,
            )
            logger.info(
                "Task TTL enabled: evicting terminal tasks older than %ds every %ds.",
                ttl_seconds,
                ttl_check_interval,
            )
        else:
            logger.warning(
                "task_ttl_seconds=%d requested, but backend=%r cannot clean up "
                "stored tasks. Use backend='file' to enable TTL eviction.",
                ttl_seconds,
                backend,
            )

    return store, cleaner
