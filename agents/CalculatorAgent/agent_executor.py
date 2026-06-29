from __future__ import annotations

from a2a.helpers import (
    new_task_from_user_message,
    new_text_artifact_update_event,
    new_text_status_update_event,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import InvalidParamsError, TaskState, TaskStatusUpdateEvent, TaskStatus
from a2a.types.a2a_pb2 import TaskState as _TaskState  # noqa: F401 (imported for type check)

from .agent import CalculatorAgent
from .logger import get_logger

logger = get_logger(__name__)


class CalculatorAgentExecutor(AgentExecutor):
    """Calculator Agent Executor."""

    def __init__(self, multi_modal: bool = False):
        self.basic_agent = CalculatorAgent()
        self.advanced_agent = CalculatorAgent(enable_advanced_tools=True)
        self._multi_modal = multi_modal

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logger.info("Agent invocation started.")

        # FIX (code smell): renamed from _validate_request — returns True when input IS empty.
        if self._is_empty_request(context):
            raise InvalidParamsError(message="Empty user input.")

        query = context.get_user_input().strip()
        agent = self._resolve_agent_for_request(context)
        logger.info(
            "Selected '%s' calculator for context_id=%s",
            "advanced" if agent is self.advanced_agent else "basic",
            context.context_id,
        )

        # FIX (risk 4): task creation and initial enqueue moved inside try so any
        # failure here is caught and emitted as a FAILED terminal event rather than
        # an unhandled exception that leaves the consumer in an unknown state.
        try:
            task = context.current_task or new_task_from_user_message(context.message)
            if not context.current_task:
                await event_queue.enqueue_event(task)

            terminal_emitted = False

            # Route to the appropriate streaming method.
            # Multi-modal: converts all Part types (text, image, JSON, URL) to
            #              LangChain content blocks via the part extractor.
            # Text-only:   uses context.get_user_input() as a plain string.
            if self._multi_modal and context.message:
                from .capabilities.multi_modal import extract_langchain_content
                content_blocks = extract_langchain_content(context.message)
                stream = agent.stream_multimodal(content_blocks, task.context_id)
            else:
                stream = agent.stream(query, task.context_id)

            async for item in stream:
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

                # Emit the final result as an artifact, then a clean terminal status.
                await event_queue.enqueue_event(
                    new_text_artifact_update_event(
                        task_id=task.id,
                        context_id=task.context_id,
                        name="calculation_result",
                        text=content,
                        last_chunk=True,
                    )
                )
                # FIX (bug 2): use TaskStatusUpdateEvent directly so the terminal
                # COMPLETED event carries NO message body.  new_text_status_update_event
                # always creates a Message with text= attached; passing "" produced a
                # spurious empty-content Message that confused clients.
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
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
            # task may not exist yet if the exception fired before the first enqueue;
            # use context ids directly so the event is always well-formed.
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id or "",
                    context_id=context.context_id or "",
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_FAILED,
                        message=_make_text_message(
                            "An internal error occurred while processing your request.",
                            context,
                        ),
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_empty_request(context: RequestContext) -> bool:
        """Returns True when the request carries no usable user text."""
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

    # FIX (bug 1): raise InvalidParamsError so on_cancel_task converts it to
    # TaskNotCancelableError (-32002).  The previous UnsupportedOperationError
    # (-32004) bypassed that conversion and gave clients the wrong semantic code.
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise InvalidParamsError(
            message="Task cancellation is not supported by this agent."
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_text_message(text: str, context: RequestContext):
    """Build a minimal text Message for use in a TaskStatus body."""
    from a2a.helpers import new_text_message
    from a2a.types import Role
    return new_text_message(
        text=text,
        role=Role.ROLE_AGENT,
        context_id=context.context_id or "",
        task_id=context.task_id or "",
    )