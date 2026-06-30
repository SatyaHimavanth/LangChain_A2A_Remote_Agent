"""Calculator Agent server entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import click
import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    DefaultServerCallContextBuilder,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.types import (
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
)
from a2a.utils.errors import InvalidRequestError
from dotenv import load_dotenv
from starlette.applications import Starlette

from .agent_executor import CalculatorAgentExecutor
from .auth import BearerTokenRequestContextBuilder, StaticTokenValidator
from .capabilities import AgentCapabilityConfig, build_server_components
from .logger import get_logger

load_dotenv()

logger = get_logger(__name__)
DEFAULT_HOST = os.getenv("CALCULATOR_AGENT_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("CALCULATOR_AGENT_PORT", "9999"))


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean capability flag from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Capability toggles: edit this block to enable or disable runtime features.
CAPABILITIES = AgentCapabilityConfig(
    push_notifications=_env_flag("CALCULATOR_ENABLE_PUSH_NOTIFICATIONS", True),
    task_management=_env_flag("CALCULATOR_ENABLE_TASK_MANAGEMENT", True),
    multi_modal=_env_flag("CALCULATOR_ENABLE_MULTI_MODAL", True),
    opaque_execution=_env_flag("CALCULATOR_ENABLE_OPAQUE_EXECUTION", True),
    task_store_backend=os.getenv("CALCULATOR_TASK_STORE_BACKEND", "file"),
    task_store_path=os.getenv("CALCULATOR_TASK_STORE_PATH", "./tasks.json"),
    task_ttl_seconds=int(os.getenv("CALCULATOR_TASK_TTL_SECONDS", "3600")),
    task_ttl_check_interval=int(os.getenv("CALCULATOR_TASK_TTL_CHECK_INTERVAL", "60")),
    push_timeout=float(os.getenv("CALCULATOR_PUSH_TIMEOUT", "10.0")),
    push_retries=int(os.getenv("CALCULATOR_PUSH_RETRIES", "3")),
    push_retry_base_delay=float(os.getenv("CALCULATOR_PUSH_RETRY_BASE_DELAY", "1.0")),
)

components = build_server_components(CAPABILITIES)


def _agent_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    """Return the JSON-RPC base URL advertised in the AgentCard."""
    return f"http://{host}:{port}"


def _bearer_requirement() -> SecurityRequirement:
    """Build the security requirement used by advanced calculator skills."""
    requirement = SecurityRequirement()
    requirement.schemes["bearerAuth"].list.append("")
    return requirement


validator = StaticTokenValidator()
request_context_builder = BearerTokenRequestContextBuilder(validator)

skill = AgentSkill(
    id="calculator_agent",
    name="Basic Calculator",
    description="Helps with basic mathematical calculations.",
    input_modes=components.input_modes,
    output_modes=components.output_modes,
    tags=["addition", "subtraction", "multiplication", "division"],
    examples=["What is 2+3*5", "What is 3-3/5*10"],
)

extended_skill = AgentSkill(
    id="super_calculator_agent",
    name="Advanced Calculator",
    description="Power and root operations for authenticated users.",
    input_modes=components.input_modes,
    output_modes=components.output_modes,
    security_requirements=[_bearer_requirement()],
    tags=["power", "root"],
    examples=["What is 2^6/8*1", "What is sqrt(4)/3*2"],
)

_security_schemes = {
    "bearerAuth": SecurityScheme(
        http_auth_security_scheme=HTTPAuthSecurityScheme(
            scheme="bearer",
            bearer_format="Opaque",
            description="Bearer token required for advanced access.",
        )
    )
}

public_agent_card = AgentCard(
    name="Calculator Agent",
    description="Basic calculator agent.",
    version="1.0.0",
    supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=_agent_url())],
    default_input_modes=components.input_modes,
    default_output_modes=components.output_modes,
    security_schemes=_security_schemes,
    skills=[skill],
)

extended_agent_card = AgentCard(
    name="Calculator Agent - Extended Edition",
    description="Full-featured calculator for authenticated users.",
    version="1.0.1",
    supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=_agent_url())],
    default_input_modes=components.input_modes,
    default_output_modes=components.output_modes,
    security_schemes=_security_schemes,
    security_requirements=[_bearer_requirement()],
    skills=[skill, extended_skill],
)

components.apply_to_card(public_agent_card)
components.apply_to_card(extended_agent_card)


async def _authenticated_extended_card(card: AgentCard, context) -> AgentCard:
    """Return the extended card only when the bearer token validates."""
    token = request_context_builder.extract_token_from_context(context)
    if token and await validator.validate(token):
        return card
    raise InvalidRequestError("A valid bearer token is required for the extended agent card.")

_executor: AgentExecutor = CalculatorAgentExecutor(
    multi_modal=CAPABILITIES.multi_modal,
)

if CAPABILITIES.opaque_execution:
    from .capabilities.opaque_execution import OpaqueAgentExecutorWrapper

    _executor = OpaqueAgentExecutorWrapper(_executor)

request_handler = DefaultRequestHandler(
    agent_executor=_executor,
    task_store=components.task_store,
    agent_card=public_agent_card,
    push_config_store=components.push_config_store,
    push_sender=components.push_sender,
    request_context_builder=request_context_builder,
    extended_agent_card=extended_agent_card,
    extended_card_modifier=_authenticated_extended_card,
)


class CalculatorServerCallContextBuilder(DefaultServerCallContextBuilder):
    """Preserve HTTP headers for bearer-token validation."""

    def build(self, request):
        context = super().build(request)
        context.state["headers"] = dict(request.headers)
        return context


server_context_builder = CalculatorServerCallContextBuilder()


@asynccontextmanager
async def lifespan(app: Starlette):
    """Start and stop background capability resources with the app."""
    logger.info("Calculator Agent starting up.")
    await components.startup()
    try:
        yield
    finally:
        await components.shutdown()
        logger.info("Calculator Agent shut down cleanly.")


routes = []
routes.extend(create_agent_card_routes(public_agent_card))
routes.extend(
    create_jsonrpc_routes(
        request_handler,
        "/",
        context_builder=server_context_builder,
        enable_v0_3_compat=True,
    )
)

app = Starlette(routes=routes, lifespan=lifespan)


@click.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True)
@click.option("--port", default=DEFAULT_PORT, show_default=True)
@click.option("--timeout", default=60, show_default=True)
def main(host: str, port: int, timeout: int) -> None:
    """Run the Calculator A2A HTTP server."""
    logger.info("Starting Calculator Agent at %s", _agent_url(host, port))
    for card in (public_agent_card, extended_agent_card):
        card.supported_interfaces[0].url = _agent_url(host, port)
    uvicorn.run(app, host=host, port=port, timeout_keep_alive=timeout)


if __name__ == "__main__":
    main()
