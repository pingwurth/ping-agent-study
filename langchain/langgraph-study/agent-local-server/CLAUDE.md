# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangGraph agent template project — a single-node graph that returns a configurable response. Designed for use with LangGraph Server and LangGraph Studio.

## Commands

```bash
# Start LangGraph dev server (required for integration tests and SDK tests)
langgraph dev

# Run unit tests
python -m pytest tests/unit_tests/

# Run integration tests (requires running server)
python -m pytest tests/integration_tests/

# Run single test file
python -m pytest tests/unit_tests/test_configuration.py

# Lint
ruff check .
ruff format . --diff

# Format
ruff format .
ruff check --select I --fix .

# Type check
mypy --strict src/
```

## Architecture

- **Entry point**: `src/agent/graph.py` — defines the `StateGraph` with `Context` (configurable params) and `State` (input/output)
- **Graph**: Single node (`call_model`) that reads `runtime.context` and returns a response
- **Server config**: `langgraph.json` maps `"agent"` → `./src/agent/graph.py:graph`
- **SDK client**: Use `langgraph_sdk.get_client(url="http://localhost:2024")` to interact with the running server

## Key Patterns

- `Context(TypedDict)` defines per-assistant configuration (set at creation or invocation time)
- `State` is a dataclass defining the graph's input/output schema
- Node functions are `async` and receive `(state, runtime)` — access config via `runtime.context`
- Graph is compiled with `StateGraph(State, context_schema=Context)` and exported as module-level `graph`
