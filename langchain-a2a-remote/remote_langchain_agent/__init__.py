"""
remote_langchain_agent
~~~~~~~~~~~~~~~~~~~~~~
LangChain-native equivalent of Google ADK's ``RemoteA2aAgent``.

Quickstart::

    from remote_langchain_agent import RemoteAgent

    research = RemoteAgent(
        name="research",
        description="Research agent",
        agent_card_url="http://localhost:8001/.well-known/agent.json",
    )

    # Simple invocation
    result = research.invoke("What is the boiling point of nitrogen?")
    print(result["messages"][-1].content)

    # Multi-turn (pass thread_id to resume the same server-side session)
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(configurable={"thread_id": "user-42"})

    result1 = research.invoke("Tell me about black holes.", config)
    result2 = research.invoke("How do they emit Hawking radiation?", config)

    # Streaming
    async for chunk in research.astream("Explain quantum entanglement."):
        if chunk.get("messages"):
            print(chunk["messages"][-1].content, end="", flush=True)

    # Use inside a LangChain supervisor via create_agent
    from langchain.agents import create_agent
    supervisor = create_agent(
        model="openai:gpt-5",
        tools=[research.as_tool()],
    )
"""

from .agent import RemoteAgent
from .exceptions import (
    A2AAuthError,
    A2AProtocolError,
    A2AStreamError,
    A2ATimeoutError,
    CardResolutionError,
    InputNormalisationError,
    RemoteAgentError,
)
from .state import ConversationStateStore, ThreadState

__all__ = [
    # Main class
    "RemoteAgent",
    # State
    "ThreadState",
    "ConversationStateStore",
    # Exceptions
    "RemoteAgentError",
    "CardResolutionError",
    "A2AProtocolError",
    "A2ATimeoutError",
    "A2AAuthError",
    "A2AStreamError",
    "InputNormalisationError",
]

__version__ = "0.1.0"
