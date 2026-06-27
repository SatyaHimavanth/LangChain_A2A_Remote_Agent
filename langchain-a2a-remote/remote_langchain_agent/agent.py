"""
agent.py
~~~~~~~~
The main public class: :class:`RemoteAgent`.

:class:`RemoteAgent` is a LangChain-native equivalent of Google ADK's
``RemoteA2aAgent``.  It communicates with any A2A-compatible agent (not just
ADK agents) and exposes the full :class:`~langchain_core.runnables.Runnable`
interface: ``invoke``, ``ainvoke``, ``stream``, ``astream``, ``batch``, and
``abatch``.

Input / output contract
~~~~~~~~~~~~~~~~~~~~~~~
**Input** (any of the following)::

    str                               → treated as a single HumanMessage
    HumanMessage                      → passed directly
    list[BaseMessage]                 → full history (last HumanMessage = query)
    {"messages": list[BaseMessage]}   → LangGraph state-dict form (preferred)

**Output**::

    {"messages": [...input_messages, AIMessage(...)]}

The state-dict output makes :class:`RemoteAgent` a drop-in LangGraph node.
For simple (non-LangGraph) use you can pull the AI reply out of
``result["messages"][-1]``.

Multi-turn sessions
~~~~~~~~~~~~~~~~~~~
Call with the same ``thread_id`` in ``RunnableConfig["configurable"]`` to
resume a server-side session.  The agent stores the A2A ``context_id``
returned on the first turn and sends it back on every subsequent turn.

Streaming
~~~~~~~~~
``stream`` / ``astream`` yield :class:`~langchain_core.messages.AIMessageChunk`
objects wrapped in ``{"messages": [chunk]}`` dicts (LangGraph convention).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import ConfigDict

from .adapters import (
    a2a_message_to_ai_message,
    a2a_task_to_ai_message,
    lc_messages_to_a2a_message,
    normalise_input,
)
from .artifacts import is_terminal_state
from .client import A2AClientManager
from .exceptions import A2AProtocolError, RemoteAgentError
from .state import ConversationStateStore, ThreadState
from .streaming import chunks_to_final_message, event_to_chunks

if TYPE_CHECKING:
    from a2a.types import Message as A2AMessage, Task as A2ATask

logger = logging.getLogger(__name__)

# Type aliases
AgentInput = Any  # str | HumanMessage | list[BaseMessage] | dict
AgentOutput = dict[str, list[BaseMessage]]  # {"messages": [...]}


class RemoteAgent(RunnableSerializable[AgentInput, AgentOutput]):
    """LangChain ``Runnable`` that proxies a remote Google ADK A2A agent.

    This is the LangChain equivalent of::

        from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

        remote = RemoteA2aAgent(
            name="research",
            description="Research agent",
            agent_card="http://localhost:8001/.well-known/agent.json",
        )

    Usage::

        from remote_langchain_agent import RemoteAgent

        research = RemoteAgent(
            name="research",
            description="Agent that answers research questions.",
            agent_card_url="http://localhost:8001/.well-known/agent.json",
        )

        result = research.invoke("What is the capital of France?")
        print(result["messages"][-1].content)

    Args:
        name: Human-readable name for this remote agent.  Used in logging
            and as the ``AIMessage`` author name.
        description: What the remote agent does.  Surfaced to supervisor
            agents when deciding which sub-agent to invoke.
        agent_card_url: Full URL to the remote agent's Agent Card JSON, e.g.
            ``http://localhost:8001/.well-known/agent.json``.
        headers: Extra HTTP headers to send with every A2A request.  Useful
            for ``Authorization`` tokens or custom tracing headers.
        timeout: Request timeout in seconds (default: 60).
        use_streaming: When ``True`` (default), use the A2A streaming
            transport.  When ``False``, collect all events and return once
            the task reaches a terminal state.
        include_chunk_metadata: Propagate A2A event metadata into
            :class:`~langchain_core.messages.AIMessageChunk`
            ``additional_kwargs`` during streaming (default: ``True``).
        httpx_client: Optional pre-configured :class:`httpx.AsyncClient`.
            The :class:`RemoteAgent` will **not** close a client it did not
            create.
    """

    # ------------------------------------------------------------------
    # Pydantic model fields (these are serialised by RunnableSerializable)
    # ------------------------------------------------------------------

    # NOTE: RunnableSerializable already declares `name: str | None = None`.
    # We override it here and also add `agent_name` as the canonical identity
    # field so the two don't conflict when the parent sets name=None by default.
    name: str | None = None  # overrides parent; used for get_name() / tracing
    agent_name: str = "remote_agent"  # the stable identity of this proxy
    description: str = ""
    agent_card_url: str
    headers: dict[str, str] = {}
    timeout: float = 60.0
    use_streaming: bool = True
    include_chunk_metadata: bool = True

    # ------------------------------------------------------------------
    # Private state (not serialised; re-created after deserialisation)
    # ------------------------------------------------------------------

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Initialise private mutable state after Pydantic construction."""
        # Mirror agent_name into name for LangSmith / tracing compatibility.
        if self.name is None:
            object.__setattr__(self, "name", self.agent_name)
        object.__setattr__(self, "_state_store", ConversationStateStore())
        object.__setattr__(
            self,
            "_client_manager",
            A2AClientManager(
                self.agent_card_url,
                headers=self.headers,
                timeout=self.timeout,
                streaming=self.use_streaming,
            ),
        )

    # ------------------------------------------------------------------
    # RunnableSerializable requirements
    # ------------------------------------------------------------------

    @property
    def InputType(self) -> type:  # noqa: N802  (LangChain convention)
        return dict

    @property
    def OutputType(self) -> type:  # noqa: N802
        return dict

    def get_name(
        self,
        suffix: Optional[str] = None,
        *,
        name: Optional[str] = None,
    ) -> str:
        label = name or self.agent_name or "RemoteAgent"
        return f"{label}{suffix}" if suffix else label

    # ------------------------------------------------------------------
    # Public Runnable interface
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: AgentInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """Invoke the remote agent synchronously.

        Safe to call from both sync and already-async contexts.  When called
        from within a running event loop (e.g. a Jupyter notebook or a FastAPI
        endpoint) it offloads to a thread executor to avoid the
        "This event loop is already running" error.

        Args:
            input: See class docstring for accepted shapes.
            config: LangChain ``RunnableConfig``.
            **kwargs: Forwarded to :meth:`ainvoke`.

        Returns:
            ``{"messages": [*original_messages, AIMessage(...)]}``
        """
        return _run_sync(self.ainvoke(input, config, **kwargs))

    async def ainvoke(
        self,
        input: AgentInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """Invoke the remote agent asynchronously.

        Args:
            input: See class docstring for accepted shapes.
            config: LangChain ``RunnableConfig``.
            **kwargs: Reserved for future use.

        Returns:
            ``{"messages": [*original_messages, AIMessage(...)]}``

        Raises:
            :class:`~remote_langchain_agent.exceptions.RemoteAgentError`:
                Any A2A-level or card-resolution error.
        """
        messages, extra_state = normalise_input(input)
        thread_id = _thread_id_from_config(config)
        state = self._state_store.get(thread_id)

        logger.debug(
            "[%s] ainvoke thread=%s context_id=%s",
            self.agent_name,
            thread_id,
            state.context_id,
        )

        a2a_msg = lc_messages_to_a2a_message(
            messages,
            context_id=state.context_id,
            task_id=state.task_id,
        )

        # Collect the full stream, then return.
        final_task: Optional["A2ATask"] = None
        final_direct_msg: Optional["A2AMessage"] = None
        all_chunks: list[AIMessageChunk] = []

        try:
            async for raw_event in self._client_manager.send_message(a2a_msg):
                chunks = event_to_chunks(
                    raw_event,
                    include_metadata=self.include_chunk_metadata,
                )
                all_chunks.extend(chunks)

                # Track the latest task so we can update state at the end.
                if isinstance(raw_event, tuple) and len(raw_event) == 2:
                    task, _ = raw_event
                    final_task = task
                else:
                    try:
                        from a2a.types import Message as _A2AMsg

                        if isinstance(raw_event, _A2AMsg):
                            final_direct_msg = raw_event
                    except ImportError:
                        pass

        except RemoteAgentError:
            raise
        except Exception as exc:
            raise A2AProtocolError(
                f"Unexpected error invoking remote agent '{self.agent_name}': {exc}"
            ) from exc

        # Update per-thread context so the next call continues the session.
        if final_task is not None:
            state.update_from_task(final_task)

        # Assemble the reply AIMessage.
        if final_task is not None:
            ai_message = a2a_task_to_ai_message(final_task)
        elif final_direct_msg is not None:
            ai_message = a2a_message_to_ai_message(final_direct_msg)
        else:
            # Rebuild from accumulated chunks as last resort.
            merged = chunks_to_final_message(all_chunks)
            ai_message = AIMessage(
                content=merged.content,
                additional_kwargs=merged.additional_kwargs,
            )

        return {**extra_state, "messages": [*messages, ai_message]}

    def stream(
        self,
        input: AgentInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[AgentOutput]:
        """Stream chunks from the remote agent synchronously.

        Yields dicts of the form ``{"messages": [AIMessageChunk(...)]}``.
        The final dict contains a complete ``AIMessage`` once the task is done.

        Internally collects the async stream via :meth:`astream` and replays
        it synchronously.  In async applications prefer :meth:`astream`.

        Args:
            input: See class docstring for accepted shapes.
            config: LangChain ``RunnableConfig``.
            **kwargs: Forwarded to :meth:`astream`.

        Yields:
            ``{"messages": [AIMessageChunk(...)]}`` during streaming, then
            ``{"messages": [...full_history, AIMessage(...)]}`` as last item.
        """

        async def _collect() -> list[AgentOutput]:
            return [item async for item in self.astream(input, config, **kwargs)]

        for item in _run_sync(_collect()):
            yield item

    async def astream(
        self,
        input: AgentInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentOutput]:
        """Stream chunks from the remote agent asynchronously.

        Each yielded value is a ``{"messages": [AIMessageChunk(...)]}`` dict
        compatible with LangGraph's streaming conventions.  After the stream
        ends, a final ``{"messages": [...full_history, AIMessage(...)]}`` dict
        is yielded so that callers collecting state always receive a complete
        view.

        Args:
            input: See class docstring for accepted shapes.
            config: LangChain ``RunnableConfig``.
            **kwargs: Reserved for future use.

        Yields:
            Streaming chunk dicts, then a final complete-state dict.
        """
        messages, extra_state = normalise_input(input)
        thread_id = _thread_id_from_config(config)
        state = self._state_store.get(thread_id)

        logger.debug(
            "[%s] astream thread=%s context_id=%s",
            self.agent_name,
            thread_id,
            state.context_id,
        )

        a2a_msg = lc_messages_to_a2a_message(
            messages,
            context_id=state.context_id,
            task_id=state.task_id,
        )

        all_chunks: list[AIMessageChunk] = []
        final_task: Optional["A2ATask"] = None
        final_direct_msg: Any = None

        try:
            async for raw_event in self._client_manager.send_message(a2a_msg):
                chunks = event_to_chunks(
                    raw_event,
                    include_metadata=self.include_chunk_metadata,
                )

                # Track task state.
                if isinstance(raw_event, tuple) and len(raw_event) == 2:
                    task, _ = raw_event
                    final_task = task
                else:
                    final_direct_msg = raw_event

                # Yield non-empty chunks immediately to the caller.
                for chunk in chunks:
                    if chunk.content:
                        all_chunks.append(chunk)
                        yield {**extra_state, "messages": [chunk]}

        except RemoteAgentError:
            raise
        except Exception as exc:
            raise A2AProtocolError(
                f"Unexpected streaming error from remote agent '{self.agent_name}': {exc}"
            ) from exc

        # Update state and emit the terminal complete-state dict.
        if final_task is not None:
            state.update_from_task(final_task)
            ai_message = a2a_task_to_ai_message(final_task)
        elif final_direct_msg is not None:
            try:
                from a2a.types import Message as A2AMessage

                if isinstance(final_direct_msg, A2AMessage):
                    ai_message = a2a_message_to_ai_message(final_direct_msg)
                else:
                    merged = chunks_to_final_message(all_chunks)
                    ai_message = AIMessage(
                        content=merged.content,
                        additional_kwargs=merged.additional_kwargs,
                    )
            except ImportError:
                merged = chunks_to_final_message(all_chunks)
                ai_message = AIMessage(
                    content=merged.content,
                    additional_kwargs=merged.additional_kwargs,
                )
        else:
            merged = chunks_to_final_message(all_chunks)
            ai_message = AIMessage(
                content=merged.content,
                additional_kwargs=merged.additional_kwargs,
            )

        yield {**extra_state, "messages": [*messages, ai_message]}

    def batch(
        self,
        inputs: list[AgentInput],
        config: Optional[RunnableConfig | list[RunnableConfig]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[AgentOutput]:
        """Invoke the remote agent for multiple inputs concurrently.

        Runs all calls concurrently in a single event loop via
        :func:`asyncio.gather`.

        Args:
            inputs: List of inputs.  Each element follows the same rules as
                :meth:`invoke`.
            config: Single config applied to all calls, or a list of configs
                one-per-input.
            return_exceptions: When ``True``, exceptions are returned in the
                results list rather than raised.
            **kwargs: Forwarded to each :meth:`ainvoke` call.

        Returns:
            List of outputs in the same order as *inputs*.
        """
        configs: list[Optional[RunnableConfig]] = _normalise_configs(config, len(inputs))

        async def _run_all() -> list[Any]:
            tasks = [
                self.ainvoke(inp, cfg, **kwargs)
                for inp, cfg in zip(inputs, configs)
            ]
            return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        return _run_sync(_run_all())

    async def abatch(
        self,
        inputs: list[AgentInput],
        config: Optional[RunnableConfig | list[RunnableConfig]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[AgentOutput]:
        """Async batch invocation.  All inputs are run concurrently.

        Args:
            inputs: List of inputs.
            config: Single config or per-input list.
            return_exceptions: Whether to swallow exceptions.
            **kwargs: Forwarded to each :meth:`ainvoke` call.

        Returns:
            List of outputs in input order.
        """
        configs = _normalise_configs(config, len(inputs))
        tasks = [
            self.ainvoke(inp, cfg, **kwargs)
            for inp, cfg in zip(inputs, configs)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return list(results)

    # ------------------------------------------------------------------
    # Agent-card inspection
    # ------------------------------------------------------------------

    async def aget_agent_card(self) -> Any:
        """Fetch (or return cached) the remote agent's Agent Card.

        Forces initialisation if not yet done.

        Returns:
            :class:`~a2a.types.AgentCard` instance.
        """
        await self._client_manager.ensure_initialised()
        return self._client_manager.agent_card

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def reset_thread(self, thread_id: Optional[str] = None) -> None:
        """Clear the stored ``context_id`` for *thread_id*.

        The next call with this thread will start a fresh server-side session.

        Args:
            thread_id: Thread to reset.  Defaults to the unnamed default thread.
        """
        self._state_store.reset(thread_id)
        logger.debug("[%s] Thread %s reset.", self.agent_name, thread_id)

    def thread_state(self, thread_id: Optional[str] = None) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the state for *thread_id*.

        Args:
            thread_id: Thread to inspect.

        Returns:
            Dict with ``context_id``, ``task_id``, ``created_at``,
            ``last_used``.
        """
        return self._state_store.get(thread_id).as_dict()

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Release underlying HTTP connections.

        Call this when the agent is no longer needed (e.g. in test teardowns).
        """
        await self._client_manager.close()

    # ------------------------------------------------------------------
    # as_tool: convenience wrapper for use with create_agent
    # ------------------------------------------------------------------

    def as_tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Any:
        """Return a LangChain :class:`~langchain_core.tools.BaseTool` wrapping this agent.

        Use this when you want to register a ``RemoteAgent`` as a tool inside
        a supervisor created with :func:`langchain.agents.create_agent`::

            supervisor = create_agent(
                model="openai:gpt-5",
                tools=[research.as_tool()],
            )

        The tool's ``_run`` / ``_arun`` methods call :meth:`invoke` /
        :meth:`ainvoke` and return the agent's text response.

        Args:
            name: Override the tool name (defaults to ``self.name``).
            description: Override the tool description (defaults to
                ``self.description``).

        Returns:
            A :class:`~langchain_core.tools.BaseTool` instance.
        """
        from langchain_core.tools import BaseTool

        agent = self
        tool_name = name or self.agent_name
        tool_description = description or self.description

        class _RemoteAgentTool(BaseTool):
            name: str = tool_name
            description: str = tool_description  # type: ignore[assignment]

            def _run(self, query: str, **kwargs: Any) -> str:
                result = agent.invoke(query, **kwargs)
                return result["messages"][-1].content

            async def _arun(self, query: str, **kwargs: Any) -> str:
                result = await agent.ainvoke(query, **kwargs)
                return result["messages"][-1].content

        return _RemoteAgentTool()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RemoteAgent(agent_name={self.agent_name!r}, "
            f"agent_card_url={self.agent_card_url!r})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _thread_id_from_config(config: Optional[RunnableConfig]) -> Optional[str]:
    """Extract ``thread_id`` from a LangChain ``RunnableConfig``, or return ``None``."""
    if config is None:
        return None
    configurable = config.get("configurable") or {}
    return configurable.get("thread_id")


def _normalise_configs(
    config: Optional[RunnableConfig | list[RunnableConfig]],
    n: int,
) -> list[Optional[RunnableConfig]]:
    """Expand a single config or list of configs to exactly *n* entries."""
    if config is None:
        return [None] * n
    if isinstance(config, list):
        if len(config) != n:
            raise ValueError(
                f"config list length ({len(config)}) must match inputs length ({n})."
            )
        return config
    return [config] * n


def _run_sync(coro: Any) -> Any:
    """Run *coro* synchronously whether or not an event loop is already running.

    * **No running loop** (normal script / test): :func:`asyncio.run` is used.
    * **Running loop** (Jupyter, FastAPI, async test): the coroutine is
      submitted to a background thread that has its own fresh event loop,
      avoiding the "This event loop is already running" error.

    Args:
        coro: An awaitable / coroutine to execute.

    Returns:
        Whatever the coroutine returns.
    """
    try:
        asyncio.get_running_loop()
        # A loop is already running — execute in a thread to avoid nesting.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop — safe to call asyncio.run directly.
        return asyncio.run(coro)
