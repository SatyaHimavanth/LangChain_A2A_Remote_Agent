# remote-langchain-agent

LangChain-native equivalent of Google ADK's `RemoteA2aAgent`.

## Quickstart

```python
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
        msg = chunk["messages"][-1]
        if reasoning := msg.additional_kwargs.get("reasoning_content"):
            print(f"[reasoning] {reasoning}")
        elif msg.content:
            print(msg.content, end="", flush=True)

# Set yield_final_state=True if you also want a final complete state dict
# after the streamed chunks.

# Non-streaming invoke returns only the final response content.
result = research.invoke("What is 2+3*5?")
print(result["messages"][-1].content)

# Use inside a LangChain supervisor via create_agent
from langchain.agents import create_agent
supervisor = create_agent(
    model="openai:gpt-5",
    tools=[research.as_tool()],
)
```
