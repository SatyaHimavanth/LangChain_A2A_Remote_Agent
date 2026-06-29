"""JSON-file-backed implementation of the A2A ``TaskStore`` interface."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from a2a.server.context import ServerCallContext
from a2a.server.owner_resolver import OwnerResolver, resolve_user_scope
from a2a.server.tasks.task_store import TaskStore
from a2a.types.a2a_pb2 import ListTasksRequest, ListTasksResponse, Task, TaskState
from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
from a2a.utils.errors import InvalidParamsError
from a2a.utils.task import decode_page_token, encode_page_token
from google.protobuf.json_format import MessageToDict, ParseDict

logger = logging.getLogger(__name__)

_TERMINAL_STATES: frozenset[int] = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)


def _task_age_seconds(task: Task) -> float | None:
    """Return task status age in seconds, or ``None`` when no timestamp exists."""
    if not task.HasField("status") or not task.status.HasField("timestamp"):
        return None
    updated_at = task.status.timestamp.ToDatetime().replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - updated_at).total_seconds()


def _task_to_dict(task: Task) -> dict:
    """Serialize a task proto to the standard A2A JSON representation."""
    return MessageToDict(task, preserving_proto_field_name=True)


def _task_from_dict(data: dict) -> Task:
    """Parse a task proto from the standard A2A JSON representation."""
    return ParseDict(data, Task())


class FileBackedTaskStore(TaskStore):
    """Persistent task store scoped by A2A owner/user identity."""

    def __init__(
        self,
        path: str | Path = "./tasks.json",
        owner_resolver: OwnerResolver = resolve_user_scope,
    ) -> None:
        self._path = Path(path)
        self._owner_resolver = owner_resolver
        self._lock = asyncio.Lock()
        self._data: dict[str, dict[str, Task]] = {}
        self._load()

    async def save(self, task: Task, context: ServerCallContext) -> None:
        """Persist or replace a task for the request owner."""
        owner = self._owner_resolver(context)
        async with self._lock:
            self._data.setdefault(owner, {})[task.id] = task
            self._flush()
        logger.debug("Task %s saved for owner=%r.", task.id, owner)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        """Return a stored task by id for the request owner."""
        owner = self._owner_resolver(context)
        async with self._lock:
            task = self._data.get(owner, {}).get(task_id)
            if task is not None:
                return task
            return self._find_task_across_owners(task_id)

    async def list(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        """List tasks using the same filters and pagination as the SDK store."""
        owner = self._owner_resolver(context)
        async with self._lock:
            owner_tasks = list(self._data.get(owner, {}).values())
            all_tasks = [
                task
                for scoped_tasks in self._data.values()
                for task in scoped_tasks.values()
            ]

        tasks = self._filter_tasks(owner_tasks, params)
        if not tasks and all_tasks != owner_tasks:
            tasks = self._filter_tasks(all_tasks, params)

        tasks.sort(
            key=lambda task: (
                task.status.HasField("timestamp")
                if task.HasField("status")
                else False,
                task.status.timestamp.ToJsonString()
                if task.HasField("status") and task.status.HasField("timestamp")
                else "",
                task.id,
            ),
            reverse=True,
        )

        total_size = len(tasks)
        start_idx = 0
        if params.page_token:
            start_task_id = decode_page_token(params.page_token)
            for index, task in enumerate(tasks):
                if task.id == start_task_id:
                    start_idx = index
                    break
            else:
                raise InvalidParamsError(f"Invalid page token: {params.page_token}")

        page_size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
        end_idx = start_idx + page_size
        next_page_token = (
            encode_page_token(tasks[end_idx].id) if end_idx < total_size else None
        )

        return ListTasksResponse(
            tasks=tasks[start_idx:end_idx],
            total_size=total_size,
            page_size=page_size,
            next_page_token=next_page_token,
        )

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        """Delete a task for the request owner, ignoring already-missing tasks."""
        owner = self._owner_resolver(context)
        async with self._lock:
            owner_tasks = self._data.get(owner, {})
            if task_id not in owner_tasks:
                logger.debug("Task %s not found for owner=%r.", task_id, owner)
                return

            del owner_tasks[task_id]
            if not owner_tasks:
                del self._data[owner]
            self._flush()
        logger.debug("Task %s deleted for owner=%r.", task_id, owner)

    async def cleanup_expired_tasks(self, ttl_seconds: int) -> int:
        """Remove terminal tasks older than ``ttl_seconds`` across all owners."""
        expired: list[tuple[str, str]] = []

        async with self._lock:
            for owner, tasks in self._data.items():
                for task_id, task in tasks.items():
                    state = task.status.state if task.HasField("status") else None
                    if state not in _TERMINAL_STATES:
                        continue
                    age = _task_age_seconds(task)
                    if age is not None and age > ttl_seconds:
                        expired.append((owner, task_id))

            for owner, task_id in expired:
                del self._data[owner][task_id]
                if not self._data[owner]:
                    del self._data[owner]

            if expired:
                self._flush()

        if expired:
            logger.info(
                "TTL cleanup evicted %d task(s) older than %ds.",
                len(expired),
                ttl_seconds,
            )
        return len(expired)

    def _load(self) -> None:
        """Load tasks from disk; invalid files are ignored with a log entry."""
        if not self._path.exists():
            logger.debug("Task store file not found at %s; starting empty.", self._path)
            return

        try:
            raw: dict[str, dict[str, dict]] = json.loads(
                self._path.read_text(encoding="utf-8")
            )
            self._data = {
                owner: {
                    task_id: _task_from_dict(task_dict)
                    for task_id, task_dict in tasks.items()
                }
                for owner, tasks in raw.items()
            }
            total = sum(len(tasks) for tasks in self._data.values())
            logger.info("Loaded %d task(s) from %s.", total, self._path)
        except Exception:
            logger.exception("Failed to load task store from %s; starting empty.", self._path)
            self._data = {}

    def _flush(self) -> None:
        """Atomically write the current in-memory task map to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {
            owner: {
                task_id: _task_to_dict(task)
                for task_id, task in tasks.items()
            }
            for owner, tasks in self._data.items()
        }

        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            delete=False,
        ) as temp_file:
            json.dump(serialized, temp_file, indent=2)
            temp_path = Path(temp_file.name)

        temp_path.replace(self._path)

    def _find_task_across_owners(self, task_id: str) -> Task | None:
        """Find a task when SDK management calls arrive without owner scope."""
        for tasks in self._data.values():
            task = tasks.get(task_id)
            if task is not None:
                return task
        return None

    @staticmethod
    def _filter_tasks(tasks: list[Task], params: ListTasksRequest) -> list[Task]:
        """Apply the SDK-compatible ListTasks filters to a task sequence."""
        filtered = tasks
        if params.context_id:
            filtered = [
                task for task in filtered if task.context_id == params.context_id
            ]

        if params.status:
            filtered = [
                task for task in filtered if task.status.state == params.status
            ]

        if params.HasField("status_timestamp_after"):
            cutoff = params.status_timestamp_after.ToJsonString()
            filtered = [
                task
                for task in filtered
                if (
                    task.HasField("status")
                    and task.status.HasField("timestamp")
                    and task.status.timestamp.ToJsonString() >= cutoff
                )
            ]

        return filtered
