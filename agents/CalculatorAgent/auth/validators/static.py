from __future__ import annotations

import hmac
import os

from ..base import AuthUser, TokenValidator
from ...logger import get_logger

logger = get_logger(__name__)


def _split_tokens(value: str | None) -> set[str]:
    return {token.strip() for token in (value or "").split(",") if token.strip()}


class StaticTokenValidator(TokenValidator):
    """Load tokens from env var or pass directly. Use only for internal tooling."""

    def __init__(self, valid_tokens: set[str] | None = None) -> None:
        env_tokens = set()
        for env_name in ("CALCULATOR_AGENT_API_TOKENS", "VALID_API_TOKENS", "VALID_API_TOKEN"):
            env_tokens.update(_split_tokens(os.getenv(env_name)))

        if valid_tokens is not None:
            self._tokens = {token.strip() for token in valid_tokens if token.strip()}
        else:
            self._tokens = env_tokens

        if self._tokens:
            logger.info("Static bearer token validator configured with %d token(s).", len(self._tokens))
        else:
            logger.warning(
                "No static bearer tokens configured; authenticated calculator tools are disabled. "
                "Set VALID_API_TOKENS or CALCULATOR_AGENT_API_TOKENS in .env."
            )

    async def validate(self, token: str) -> AuthUser | None:
        for valid in self._tokens:
            if hmac.compare_digest(token.encode(), valid.encode()):
                return AuthUser(user_id=f"static:{token[:8]}")
        return None
