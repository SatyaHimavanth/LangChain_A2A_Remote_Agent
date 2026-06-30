## A2A Calculator Agent

This project contains independent A2A agents under `agents/`. Agent-specific code,
including logging and auth helpers, lives inside each agent folder. Root-level files
are project tools such as `test_all_client.py` and `web_dashboard_server.py`.

## Setup

```bash
uv sync
```

Create a `.env` file with your model settings. For advanced calculator access with
the static bearer-token validator, add this exact token list and restart the server:

```bash
VALID_API_TOKENS=local-dev-token
```

You can also use `CALCULATOR_AGENT_API_TOKENS` for calculator-specific tokens.

## Run Calculator Agent

```bash
uv run -m agents.CalculatorAgent --host 127.0.0.1 --port 9999
```

## CLI Test Client

```bash
# Run every capability check.
uv run python test_all_client.py --agent http://127.0.0.1:9999 --token local-dev-token

# Pick checks interactively.
uv run python test_all_client.py --select --token local-dev-token

# Run selected checks only.
uv run python test_all_client.py --tests card,chat,stream,tasks --token local-dev-token

# Start the local CalculatorAgent, wait for it, then run the tests.
uv run python test_all_client.py --start-agent --token local-dev-token
```

Available checks: `card`, `chat`, `stream`, `extended`, `tasks`, `multimodal`, `push`.

## Web Dashboard

```bash
uv run python web_dashboard_server.py --host 127.0.0.1 --port 8766
```

- Open `http://127.0.0.1:8766`, enter the A2A URL and optional bearer token, then connect.
- The dashboard supports chat, stream/invoke mode, automatic push registration, file input
- When multimodal modes are advertised, task inspection, task subscription, cancellation,
- Push notification feed, agent-card inspection, and light/dark mode.
- To surface those tool-by-tool steps in the timeline, disable opaque mode and restart the agent. Add this to your .env:
```bash
CALCULATOR_ENABLE_OPAQUE_EXECUTION=false
```
- Then the timeline will additionally show:
```bash
● WORKING   Invoking addition tool with {'a': 2, 'b': 3}   state=WORKING · is_task_complete=false
● WORKING   Response from addition tool is 5                state=WORKING · is_task_complete=false
```
## Reference Samples

- https://github.com/a2aproject/a2a-samples/tree/main/samples/python
