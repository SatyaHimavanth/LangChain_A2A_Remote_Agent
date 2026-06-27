"""
examples.py
~~~~~~~~~~~
Complete, self-contained usage examples for remote_langchain_agent.

Prerequisites
~~~~~~~~~~~~~
1.  An ADK A2A server running locally::

        pip install "google-adk[a2a]"
        adk api_server --a2a --port 8001 ./my_agents/

2.  The remote agent must have an agent card at::

        http://localhost:8001/.well-known/agent.json

    (or a path like ``/a2a/my_agent/.well-known/agent.json`` if hosted under
    a prefix — see the ``agent_card_url`` examples below).

Run a specific example::

    python examples.py simple
    python examples.py multi_turn
    python examples.py streaming
    python examples.py create_agent_example
    python examples.py batch
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

AGENT_CARD_URL = "http://localhost:8001/.well-known/agent.json"


def _build_agent(name: str = "research", url: str = AGENT_CARD_URL):
    """Helper: create a RemoteAgent instance."""
    from remote_langchain_agent import RemoteAgent

    return RemoteAgent(
        name=name,
        description="A remote ADK agent accessible via the A2A protocol.",
        agent_card_url=url,
        timeout=30.0,
    )


# ---------------------------------------------------------------------------
# Example 1: Simple invocation
# ---------------------------------------------------------------------------


def example_simple_invoke():
    """
    Basic invoke — treats the RemoteAgent like any LangChain Runnable.
    """
    from remote_langchain_agent import RemoteAgent

    agent = RemoteAgent(
        name="prime_checker",
        description="Checks if a number is prime.",
        agent_card_url="http://localhost:8001/.well-known/agent.json",
    )

    # Accepts a plain string.
    result = agent.invoke("Is 97 a prime number?")

    # The output is always {"messages": [..., AIMessage]}.
    ai_reply = result["messages"][-1]
    print("Agent replied:", ai_reply.content)
    print("A2A task ID  :", ai_reply.additional_kwargs.get("a2a_task_id"))


# ---------------------------------------------------------------------------
# Example 2: Async invocation
# ---------------------------------------------------------------------------


async def example_ainvoke():
    """Async variant — preferred in async applications."""
    from remote_langchain_agent import RemoteAgent

    agent = RemoteAgent(
        name="research",
        description="Research questions.",
        agent_card_url=AGENT_CARD_URL,
    )

    result = await agent.ainvoke("What is the boiling point of nitrogen?")
    print(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# Example 3: Multi-turn conversation
# ---------------------------------------------------------------------------


async def example_multi_turn():
    """
    Multi-turn conversation using thread_id to maintain server-side context.

    The agent stores the A2A context_id returned on the first turn and
    re-sends it on every subsequent turn so the remote agent can resume the
    same session.
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableConfig

    from remote_langchain_agent import RemoteAgent

    agent = RemoteAgent(
        name="research",
        description="Research agent.",
        agent_card_url=AGENT_CARD_URL,
    )

    # thread_id scopes the conversation.  Any string works.
    thread_cfg = RunnableConfig(configurable={"thread_id": "demo-conversation-1"})

    # Turn 1.
    r1 = await agent.ainvoke(
        {"messages": [HumanMessage("Tell me about black holes.")]},
        thread_cfg,
    )
    print("Turn 1:", r1["messages"][-1].content)

    # Turn 2 — the agent now knows we're asking a follow-up.
    r2 = await agent.ainvoke(
        {"messages": [HumanMessage("How do they emit Hawking radiation?")]},
        thread_cfg,
    )
    print("Turn 2:", r2["messages"][-1].content)

    # Inspect the stored A2A context_id.
    print("State:", agent.thread_state("demo-conversation-1"))

    # Reset to start fresh on the next call.
    agent.reset_thread("demo-conversation-1")


# ---------------------------------------------------------------------------
# Example 4: Streaming
# ---------------------------------------------------------------------------


async def example_streaming():
    """
    Stream tokens from the remote agent as they arrive.

    Intermediate items carry AIMessageChunk objects; the final item carries a
    complete AIMessage together with the full message history.
    """
    from remote_langchain_agent import RemoteAgent

    agent = RemoteAgent(
        name="research",
        description="Research agent.",
        agent_card_url=AGENT_CARD_URL,
        include_chunk_metadata=True,
    )

    print("Streaming response:")
    async for item in agent.astream("Explain quantum entanglement in simple terms."):
        msgs = item.get("messages", [])
        if not msgs:
            continue
        last = msgs[-1]
        # Intermediate chunks arrive as AIMessageChunk.
        from langchain_core.messages import AIMessageChunk

        if isinstance(last, AIMessageChunk) and last.content:
            print(last.content, end="", flush=True)
        else:
            # Final complete-state dict — print a newline and stop.
            print()
            print("\n--- Final AIMessage ---")
            print(last.content)
            print("Metadata:", last.additional_kwargs.get("a2a_task_id"))


# ---------------------------------------------------------------------------
# Example 5: Using RemoteAgent inside create_agent (supervisor pattern)
# ---------------------------------------------------------------------------


def example_create_agent():
    """
    Use RemoteAgent as a tool inside a LangChain supervisor agent.

    create_agent expects a list of tools.  RemoteAgent.as_tool() wraps the
    remote agent as a BaseTool so it can be registered alongside local tools.
    The supervisor LLM decides when to invoke the remote agent based on its
    description.
    """
    from langchain.agents import create_agent
    from langchain.tools import tool

    from remote_langchain_agent import RemoteAgent

    # The remote agent we want to call.
    prime_agent = RemoteAgent(
        name="prime_checker",
        description=(
            "Checks whether a given integer is a prime number. "
            "Input should be a single integer as a string, e.g. '97'."
        ),
        agent_card_url="http://localhost:8001/.well-known/agent.json",
    )

    # A local tool for comparison.
    @tool
    def add_numbers(expression: str) -> str:
        """Add two numbers.  Input should be like '3 + 4'."""
        a, _, b = expression.partition("+")
        return str(int(a.strip()) + int(b.strip()))

    # Build the supervisor.
    supervisor = create_agent(
        model="openai:gpt-5",          # any supported provider
        tools=[
            add_numbers,
            prime_agent.as_tool(),     # remote agent registered as a tool
        ],
        system_prompt=(
            "You are a helpful assistant. "
            "Use the prime_checker tool for primality questions and "
            "add_numbers for addition."
        ),
    )

    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": "Is 127 prime, and what is 15 + 28?"}]}
    )
    print(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# Example 6: RemoteAgent as a LangGraph node (multi-agent orchestration)
# ---------------------------------------------------------------------------


async def example_langgraph_node():
    """
    Use RemoteAgent directly as a node inside a LangGraph StateGraph.

    This is more powerful than the as_tool() approach because the remote agent
    has access to the full message history and returns a proper state update.
    """
    from typing import Annotated, TypedDict

    from langchain_core.messages import BaseMessage
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages

    from remote_langchain_agent import RemoteAgent

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    remote = RemoteAgent(
        name="summariser",
        description="Summarises long documents.",
        agent_card_url="http://localhost:8001/.well-known/agent.json",
    )

    # The remote agent's ainvoke signature matches what LangGraph expects for
    # a node: takes state dict, returns state dict.
    graph = StateGraph(AgentState)
    graph.add_node("remote_summariser", remote.ainvoke)
    graph.set_entry_point("remote_summariser")
    graph.add_edge("remote_summariser", END)

    app = graph.compile()

    result = await app.ainvoke(
        {"messages": [{"role": "user", "content": "Summarise the A2A protocol spec."}]}
    )
    print(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# Example 7: Batch processing
# ---------------------------------------------------------------------------


async def example_batch():
    """
    Run multiple independent queries concurrently.

    All calls are dispatched concurrently via asyncio.gather so the total
    wall-clock time is roughly the time for the slowest single call.
    """
    from remote_langchain_agent import RemoteAgent

    agent = RemoteAgent(
        name="classifier",
        description="Classifies text.",
        agent_card_url=AGENT_CARD_URL,
    )

    inputs = [
        "Is 2 prime?",
        "Is 4 prime?",
        "Is 7 prime?",
    ]

    results = await agent.abatch(inputs)
    for inp, res in zip(inputs, results):
        print(f"Q: {inp!r:20s}  →  {res['messages'][-1].content}")


# ---------------------------------------------------------------------------
# Example 8: Custom authentication (Bearer token, Google Cloud)
# ---------------------------------------------------------------------------


async def example_auth():
    """
    Pass authorisation headers to the remote agent.

    Useful when the A2A server is deployed behind a Google Cloud load balancer
    or any other OAuth2-protected endpoint.
    """
    import os

    from remote_langchain_agent import RemoteAgent

    # Option A: static bearer token.
    agent_static = RemoteAgent(
        name="secure_agent",
        description="Agent behind auth.",
        agent_card_url="https://my-agent.cloud.run/a2a/.well-known/agent.json",
        headers={"Authorization": f"Bearer {os.environ['MY_API_TOKEN']}"},
    )

    # Option B: supply your own httpx.AsyncClient with a Google Auth transport.
    import httpx

    # from google.auth.transport.requests import Request as GRequest
    # from google.oauth2 import service_account
    # credentials = service_account.Credentials.from_service_account_file(
    #     "key.json",
    #     scopes=["https://www.googleapis.com/auth/cloud-platform"],
    # )
    # credentials.refresh(GRequest())
    # auth_client = httpx.AsyncClient(
    #     headers={"Authorization": f"Bearer {credentials.token}"}
    # )

    # agent_gcp = RemoteAgent(
    #     name="gcp_agent",
    #     description="Agent Engine agent.",
    #     agent_card_url="https://...",
    #     httpx_client=auth_client,
    # )

    print("Auth example shown (live call skipped in demo mode).")


# ---------------------------------------------------------------------------
# Example 9: Agent Card inspection
# ---------------------------------------------------------------------------


async def example_inspect_card():
    """Fetch the remote agent's Agent Card to inspect its capabilities."""
    from remote_langchain_agent import RemoteAgent

    agent = RemoteAgent(
        name="inspect_demo",
        description="Demo",
        agent_card_url=AGENT_CARD_URL,
    )

    card = await agent.aget_agent_card()
    print("Agent name       :", card.name)
    print("Agent description:", card.description)
    if hasattr(card, "capabilities"):
        print("Capabilities     :", card.capabilities)


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------


EXAMPLES = {
    "simple": example_simple_invoke,
    "ainvoke": example_ainvoke,
    "multi_turn": example_multi_turn,
    "streaming": example_streaming,
    "create_agent_example": example_create_agent,
    "langgraph_node": example_langgraph_node,
    "batch": example_batch,
    "auth": example_auth,
    "inspect_card": example_inspect_card,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "simple"
    fn = EXAMPLES.get(name)
    if fn is None:
        print(f"Unknown example '{name}'.  Available: {list(EXAMPLES)}")
        sys.exit(1)

    if asyncio.iscoroutinefunction(fn):
        asyncio.run(fn())
    else:
        fn()
