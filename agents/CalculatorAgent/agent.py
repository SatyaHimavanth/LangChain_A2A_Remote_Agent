"""LangChain-backed calculator logic used by the A2A executor."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from .agent_tools import addition, division, multiplication, power, root, subtraction
from .llms import llm
from .logger import get_logger

logger = get_logger(__name__)


class ResponseFormat(BaseModel):
    """Structured response contract returned by the LangChain agent."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class CalculatorAgent:
    """Calculator agent that streams tool progress and a final structured answer."""

    SYSTEM_INSTRUCTION = (
        "You are a helpful, precise, and reliable mathematical assistant. Your primary "
        "function is to solve arithmetic and mathematical problems accurately. Use only "
        "the tools that are available to you to perform calculations. Determine whether "
        "the required operation can be completed using the available tools before "
        "answering. You may combine multiple tool calls to solve complex expressions "
        "while respecting the standard order of operations. Do not perform calculations "
        "from your own knowledge or simulate unavailable tools. If the required "
        "calculation cannot be completed because the necessary tool or operation is not "
        "available, respond with a polite rejection explaining that you cannot complete "
        "the request because the required tool or operation is not available."
    )
    ADVANCED_SYSTEM_INSTRUCTION = (
        f"{SYSTEM_INSTRUCTION} The caret symbol ^ means exponentiation. For expressions "
        "like 2^6, use the power tool with base=2 and exponent=6 before applying "
        "division, multiplication, addition, or subtraction."
    )

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, enable_advanced_tools: bool = False):
        self.model = llm
        self.tools = [addition, subtraction, multiplication, division]
        if enable_advanced_tools:
            self.tools.extend([power, root])

        # MemorySaver keeps conversation/tool state isolated by A2A context_id.
        self._memory = MemorySaver()
        self.graph = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=self._memory,
            system_prompt=(
                self.ADVANCED_SYSTEM_INSTRUCTION
                if enable_advanced_tools
                else self.SYSTEM_INSTRUCTION
            ),
            response_format=ToolStrategy(ResponseFormat),
        )

    async def stream(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream progress dictionaries for a plain-text calculator request."""
        inputs = {"messages": [("user", query)]}
        async for item in self._stream_graph(inputs, context_id):
            yield item

    async def stream_multimodal(
        self,
        content: list[dict],
        context_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream progress for modality-agnostic LangChain content blocks."""
        inputs = {"messages": [HumanMessage(content=content)]}
        async for item in self._stream_graph(inputs, context_id):
            yield item

    async def _stream_graph(
        self,
        inputs: dict[str, Any],
        context_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Run the LangGraph agent and normalize each update for the A2A layer."""
        config = {"configurable": {"thread_id": context_id}}
        logger.info("Agent invoked with thread_id=%s", context_id)

        async for chunk in self.graph.astream(
            inputs,
            config=config,
            stream_mode="updates",
        ):
            logger.info("Agent response chunk: %s", chunk)
            async for item in self._process_chunk(chunk):
                yield item

    async def _process_chunk(
        self,
        chunk: dict,
    ) -> AsyncIterable[dict[str, Any]]:
        """Convert LangGraph updates into status dictionaries for AgentExecutor."""
        for _, data in chunk.items():
            structured_response = data.get("structured_response")
            if structured_response is not None:
                yield self.get_agent_response(structured_response)
                continue

            messages = data.get("messages")
            if not messages:
                continue

            message = messages[-1]
            if isinstance(message, AIMessage):
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = str(tool_call.get("args", {}))
                        yield {
                            "status": "working",
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": (
                                f"Invoking {tool_name} tool with arguments {tool_args}"
                            ),
                        }
                    continue

                content = self._message_content(message)
                has_content = bool(content.strip())
                logger.warning(
                    "AIMessage without tool_calls and without structured_response; "
                    "treating as %s.",
                    "completed" if has_content else "input_required",
                )
                yield {
                    "status": "completed" if has_content else "input_required",
                    "is_task_complete": has_content,
                    "require_user_input": not has_content,
                    "content": content or "Please provide more details.",
                }
                continue

            if isinstance(message, ToolMessage):
                tool_name = message.name or "tool"
                tool_response = self._message_content(message)
                yield {
                    "status": "working",
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"Response from {tool_name} tool is {tool_response}",
                }

    def get_agent_response(self, structured_response: Any) -> dict[str, Any]:
        """Map the final structured model response to executor status fields."""
        logger.info("Final agent response: %s", structured_response)

        payload: ResponseFormat | None
        if isinstance(structured_response, ResponseFormat):
            payload = structured_response
        elif isinstance(structured_response, dict):
            try:
                payload = ResponseFormat.model_validate(structured_response)
            except Exception:
                payload = None
        else:
            payload = None

        if payload and payload.status == "input_required":
            return {
                "status": "input_required",
                "is_task_complete": False,
                "require_user_input": True,
                "content": payload.message,
            }
        if payload and payload.status == "error":
            return {
                "status": "error",
                "is_task_complete": False,
                "require_user_input": False,
                "content": payload.message,
            }
        if payload and payload.status == "completed":
            return {
                "status": "completed",
                "is_task_complete": True,
                "require_user_input": False,
                "content": payload.message,
            }

        return {
            "status": "error",
            "is_task_complete": False,
            "require_user_input": False,
            "content": "We are unable to process your request at the moment. Please try again.",
        }

    @staticmethod
    def _message_content(message: Any) -> str:
        """Return message content as displayable text for progress updates."""
        content = getattr(message, "content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)
