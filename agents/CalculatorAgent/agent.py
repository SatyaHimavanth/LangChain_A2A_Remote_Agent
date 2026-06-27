from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from .agent_tools import addition, division, multiplication, power, root, subtraction
from .logger import get_logger
from .llm_models import llm

logger = get_logger(__name__)

memory = MemorySaver()


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class CalculatorAgent:
    """Calculator Agent."""

    SYSTEM_INSTRUCTION = (
        "You are a helpful, precise, and reliable mathematical assistant. "
        "Your primary function is to solve arithmetic and mathematical problems accurately. "
        "Use only the tools that are available to you to perform calculations. "
        "Determine whether the required operation can be completed using the available tools before answering. "
        "You may combine multiple tool calls to solve complex expressions while respecting the standard order of operations. "
        "Do not perform calculations from your own knowledge or simulate unavailable tools. "
        "If the required calculation cannot be completed because the necessary tool is unavailable, "
        "respond with a polite rejection explaining that you cannot complete the request because the required tool or operation is not available."
    )

    ADVANCED_SYSTEM_INSTRUCTION = (
        f"{SYSTEM_INSTRUCTION} The caret symbol ^ means exponentiation. "
        "For expressions like 2^6, use the power tool with base=2 and exponent=6 "
        "before applying division, multiplication, addition, or subtraction."
    )

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide more information to complete the request. "
        "Set response status to error if there is an error while processing the request. "
        "Set response status to completed if the request is complete."
    )

    def __init__(self, enable_advanced_tools: bool = False):
        self.model = llm

        self.tools = [addition, subtraction, multiplication, division]
        if enable_advanced_tools:
            self.tools.extend([power, root])

        self.graph = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            system_prompt=(
                self.ADVANCED_SYSTEM_INSTRUCTION
                if enable_advanced_tools
                else self.SYSTEM_INSTRUCTION
            ),
            response_format=ToolStrategy(ResponseFormat),
        )

    async def stream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": context_id}}
        logger.info("Agent invoked with thread_id=%s", context_id)

        async for chunk in self.graph.astream(inputs, config=config, stream_mode="updates"):
            logger.info("Agent response chunk: %s", chunk)

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
                        for tool in message.tool_calls:
                            tool_name = tool.get("name", "unknown")
                            tool_args = str(tool.get("args", {}))
                            yield {
                                "status": "working",
                                "is_task_complete": False,
                                "require_user_input": False,
                                "content": f"Invoking {tool_name} tool with arguments {tool_args}",
                            }
                    else:
                        content = self._message_content(message)
                        has_content = bool(content.strip())
                        yield {
                            "status": "completed" if has_content else "input_required",
                            "is_task_complete": has_content,
                            "require_user_input": not has_content,
                            "content": content or "Please provide more details.",
                        }

                elif isinstance(message, ToolMessage):
                    tool_name = message.name or "tool"
                    tool_response = self._message_content(message)
                    yield {
                        "status": "working",
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"Response from {tool_name} tool is {tool_response}",
                    }

    def get_agent_response(self, structured_response: Any) -> dict[str, Any]:
        logger.info("Final agent response: %s", structured_response)
        logger.info("Final response type: %s", type(structured_response))

        if isinstance(structured_response, ResponseFormat):
            payload = structured_response
        elif isinstance(structured_response, dict):
            try:
                payload = ResponseFormat.model_validate(structured_response)
            except Exception:
                payload = None
        else:
            payload = None

        if payload is not None:
            if payload.status == "input_required":
                return {
                    "status": "input_required",
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": payload.message,
                }
            if payload.status == "error":
                return {
                    "status": "error",
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": payload.message,
                }
            if payload.status == "completed":
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
        content = getattr(message, "content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
