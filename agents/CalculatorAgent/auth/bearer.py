from __future__ import annotations

from a2a.auth.user import User
from a2a.server.agent_execution import RequestContext, RequestContextBuilder
from a2a.server.context import ServerCallContext
from a2a.types import SendMessageRequest, Task

from .base import TokenValidator
from ..logger import get_logger

logger = get_logger(__name__)


class AuthenticatedUser(User):
    def __init__(self, user_name: str, user_id: str | None = None) -> None:
        self._user_name = user_name
        self.user_id = user_id or user_name

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self._user_name


class BearerTokenRequestContextBuilder(RequestContextBuilder):
    def __init__(self, validator: TokenValidator) -> None:
        self._validator = validator

    async def build(
        self,
        context: ServerCallContext,
        params: SendMessageRequest | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
    ) -> RequestContext:
        token = self.extract_token_from_context(context)

        if token:
            user = await self._validator.validate(token)
            if user:
                logger.info("Authenticated request for user_id=%s", user.user_id)
                return RequestContext(
                    call_context=ServerCallContext(
                        state={
                            "auth_token_valid": True,
                            "auth_user_id": user.user_id,
                            "email": user.email,
                            "scopes": user.scopes,
                            "raw_claims": user.raw_claims,
                        },
                        user=AuthenticatedUser(user_name=user.user_id, user_id=user.user_id),
                        tenant=context.tenant,
                        requested_extensions=set(context.requested_extensions),
                    ),
                    request=params,
                    task_id=task_id,
                    context_id=context_id,
                    task=task,
                )

            logger.info("Bearer token present but validation failed.")
        else:
            logger.info("No bearer token present; creating anonymous request context.")

        return RequestContext(
            call_context=ServerCallContext(
                state={"auth_token_valid": False},
                user=context.user,
                tenant=context.tenant,
                requested_extensions=set(context.requested_extensions),
            ),
            request=params,
            task_id=task_id,
            context_id=context_id,
            task=task,
        )

    @staticmethod
    def extract_token_from_context(context: ServerCallContext) -> str | None:
        headers = context.state.get("headers", {})
        header = ""
        if isinstance(headers, dict):
            header = str(headers.get("authorization", "")).strip()
        if not header:
            return None

        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None

        return token.strip()
