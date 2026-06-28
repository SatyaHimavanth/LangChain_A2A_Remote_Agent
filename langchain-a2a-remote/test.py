from remote_langchain_agent import RemoteAgent

research = RemoteAgent(
    name="calculator",
    description="Calculator agent",
    agent_card_url="http://localhost:9999/.well-known/agent-card.json",
)

# Simple invocation
result = research.invoke("What is 2+4 / 3?")
print(result["messages"][-1].content)

# Multi-turn (pass thread_id to resume the same server-side session)
from langchain_core.runnables import RunnableConfig
config = RunnableConfig(configurable={"thread_id": "user-42"})

result1 = research.invoke("Tell me about black holes.", config)
result2 = research.invoke("How do they emit Hawking radiation?", config)
print(result2["messages"][-1].content)

# Streaming
for chunk in research.stream("Explain quantum entanglement."):
    if chunk.get("messages"):
        print(chunk["messages"][-1].content, end="", flush=True)
