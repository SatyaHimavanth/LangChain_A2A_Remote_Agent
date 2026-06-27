## A2A Calculator Agent

This project contains independent A2A agents under `agents/`. Agent-specific code,
including logging and auth helpers, lives inside each agent folder. Keep root-level
files for project tooling such as `test_client.py`.

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

## Test The Running Agent

In another terminal:

```bash
uv run python test_client.py --url http://127.0.0.1:9999 --message "What is 2+3*5?"
uv run python test_client.py --url http://127.0.0.1:9999 --message "What is 2+3*5?" --response-type stream
uv run python test_client.py --url http://127.0.0.1:9999 --message "What is 2^6/8*1?" --token local-dev-token --timeout 180
```

## Reference Samples

- https://github.com/a2aproject/a2a-samples/tree/main/samples/python

