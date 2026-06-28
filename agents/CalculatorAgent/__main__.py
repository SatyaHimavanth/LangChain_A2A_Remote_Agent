from __future__ import annotations

import os

import click
import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    DefaultServerCallContextBuilder,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
)
from dotenv import load_dotenv
from starlette.applications import Starlette

from .agent import CalculatorAgent
from .agent_executor import CalculatorAgentExecutor
from .auth import BearerTokenRequestContextBuilder, StaticTokenValidator
from .logger import get_logger

load_dotenv()

logger = get_logger(__name__)
DEFAULT_HOST = os.getenv("CALCULATOR_AGENT_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("CALCULATOR_AGENT_PORT", "9999"))


def _agent_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"


def _bearer_requirement() -> SecurityRequirement:
    requirement = SecurityRequirement()
    requirement.schemes["bearerAuth"].list.append("")
    return requirement


# --- Auth setup (swap validator here, nothing else changes) ---
validator = StaticTokenValidator()
# validator = OAuthIntrospectionValidator(
#     introspection_url=os.environ["OAUTH_INTROSPECT_URL"],
#     client_id=os.environ["OAUTH_CLIENT_ID"],
#     client_secret=os.environ["OAUTH_CLIENT_SECRET"],
# )
request_context_builder = BearerTokenRequestContextBuilder(validator)

# --- Skills ---
skill = AgentSkill(
    id="calculator_agent",
    name="Basic Calculator",
    description="Helps with basic mathematical calculations",
    input_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
    output_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
    tags=["addition", "subtraction", "multiplication", "division"],
    examples=["What is 2+3*5", "What is 3-3/5*10"],
)

extended_skill = AgentSkill(
    id="super_calculator_agent",
    name="Advanced Calculator",
    description="Power and root operations for authenticated users.",
    input_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
    output_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
    security_requirements=[_bearer_requirement()],
    tags=["power", "root"],
    examples=["What is 2^6/8*1", "What is sqrt(4)/3*2"],
)

# --- Cards ---
public_agent_card = AgentCard(
    name="Calculator Agent",
    description="Basic calculator agent",
    version="1.0.0",
    supported_interfaces=[
        AgentInterface(protocol_binding="JSONRPC", url=_agent_url()),
    ],
    default_input_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
    default_output_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
    capabilities=AgentCapabilities(
        streaming=True,
        push_notifications=True,
        extended_agent_card=True,
    ),
    security_schemes={
        "bearerAuth": SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                scheme="bearer",
                bearer_format="Opaque",
                description="Bearer token required for advanced access.",
            )
        )
    },
    skills=[skill],
)

extended_agent_card = AgentCard(
    name="Calculator Agent - Extended Edition",
    description="Full-featured calculator for authenticated users.",
    version="1.0.1",
    supported_interfaces=[
        AgentInterface(protocol_binding="JSONRPC", url=_agent_url()),
    ],
    default_input_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
    default_output_modes=CalculatorAgent.SUPPORTED_CONTENT_TYPES,
    capabilities=AgentCapabilities(
        streaming=True,
        push_notifications=True,
        extended_agent_card=True,
    ),
    security_schemes=public_agent_card.security_schemes,
    security_requirements=[_bearer_requirement()],
    skills=[skill, extended_skill],
)

# --- Request handler ---
request_handler = DefaultRequestHandler(
    agent_executor=CalculatorAgentExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=public_agent_card,
    push_config_store=InMemoryPushNotificationConfigStore(),
    request_context_builder=request_context_builder,
    extended_agent_card=extended_agent_card,
)


class CalculatorServerCallContextBuilder(DefaultServerCallContextBuilder):
    """Preserves HTTP headers for bearer-token validation."""

    def build(self, request):
        context = super().build(request)
        context.state["headers"] = dict(request.headers)
        return context


server_context_builder = CalculatorServerCallContextBuilder()

# --- Routes ---
routes = []
routes.extend(create_agent_card_routes(public_agent_card))
routes.extend(create_jsonrpc_routes(request_handler, "/", context_builder=server_context_builder, enable_v0_3_compat=True))

app = Starlette(routes=routes)


@click.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True)
@click.option("--port", default=DEFAULT_PORT, show_default=True)
@click.option("--timeout", default=60, show_default=True)
def main(host: str, port: int, timeout: int) -> None:
    logger.info("Starting Calculator Agent at %s", _agent_url(host, port))
    for card in (public_agent_card, extended_agent_card):
        card.supported_interfaces[0].url = _agent_url(host, port)
    uvicorn.run(app, host=host, port=port, timeout_keep_alive=timeout)


if __name__ == "__main__":
    main()
