from __future__ import annotations

from a2a.helpers import (
    new_task_from_user_message,
    new_text_artifact_update_event,
    new_text_status_update_event,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import InvalidParamsError, TaskState, UnsupportedOperationError

from .agent import CalculatorAgent
from .logger import get_logger

logger = get_logger(__name__)


class CalculatorAgentExecutor(AgentExecutor):
    """Calculator Agent Executor."""

    def __init__(self):
        self.basic_agent = CalculatorAgent()
        self.advanced_agent = CalculatorAgent(enable_advanced_tools=True)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logger.info("Agent invocation started.")

        if self._validate_request(context):
            raise InvalidParamsError()

        query = context.get_user_input().strip()
        if not query:
            raise InvalidParamsError(message="Empty user input.")

        agent = self._resolve_agent_for_request(context)
        logger.info(
            "Selected '%s' calculator for context_id=%s",
            "advanced" if agent is self.advanced_agent else "basic",
            context.context_id,
        )

        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        try:
            terminal_emitted = False

            async for item in agent.stream(query, task.context_id):
                logger.info("Agent stream item: %s", item)
                is_task_complete = item.get("is_task_complete", False)
                require_user_input = item.get("require_user_input", False)
                response_status = item.get("status", "")
                content = item.get("content", "")

                if not is_task_complete and not require_user_input and response_status != "error":
                    await event_queue.enqueue_event(
                        new_text_status_update_event(
                            task_id=task.id,
                            context_id=task.context_id,
                            state=TaskState.TASK_STATE_WORKING,
                            text=content,
                        )
                    )
                    continue

                if response_status == "error":
                    await event_queue.enqueue_event(
                        new_text_status_update_event(
                            task_id=task.id,
                            context_id=task.context_id,
                            state=TaskState.TASK_STATE_FAILED,
                            text=content,
                        )
                    )
                    terminal_emitted = True
                    break

                if require_user_input:
                    await event_queue.enqueue_event(
                        new_text_status_update_event(
                            task_id=task.id,
                            context_id=task.context_id,
                            state=TaskState.TASK_STATE_INPUT_REQUIRED,
                            text=content,
                        )
                    )
                    terminal_emitted = True
                    break

                await event_queue.enqueue_event(
                    new_text_artifact_update_event(
                        task_id=task.id,
                        context_id=task.context_id,
                        name="calculation_result",
                        text=content,
                        last_chunk=True,
                    )
                )
                await event_queue.enqueue_event(
                    new_text_status_update_event(
                        task_id=task.id,
                        context_id=task.context_id,
                        state=TaskState.TASK_STATE_COMPLETED,
                        text="",
                    )
                )
                terminal_emitted = True
                break

            if not terminal_emitted:
                await event_queue.enqueue_event(
                    new_text_status_update_event(
                        task_id=task.id,
                        context_id=task.context_id,
                        state=TaskState.TASK_STATE_FAILED,
                        text="Agent stream ended without producing a terminal event.",
                    )
                )

        except Exception:
            logger.exception("Error during agent stream")
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_FAILED,
                    text="An internal error occurred while processing your request.",
                )
            )

    def _validate_request(self, context: RequestContext) -> bool:
        return not bool(context.get_user_input().strip())

    def _resolve_agent_for_request(self, context: RequestContext) -> CalculatorAgent:
        if self._is_request_authenticated(context):
            return self.advanced_agent
        return self.basic_agent

    @staticmethod
    def _is_request_authenticated(context: RequestContext) -> bool:
        call_context = context.call_context
        if not call_context.user.is_authenticated:
            return False
        return bool(call_context.state.get("auth_token_valid"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()
